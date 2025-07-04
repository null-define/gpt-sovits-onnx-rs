use std::{collections::LinkedList, fmt::Debug, sync::Arc};

use anyhow::{Context, Result};
use jieba_rs::Jieba;
use log::{debug, info, warn};
use ndarray::{Array2, Axis, concatenate};
use once_cell::sync::Lazy;
use regex::Regex; // Removed 'Split' as it's no longer needed

use crate::text::{
    bert::BertModel,
    en::{EnSentence, EnWord, g2p_en::G2pEn},
    num::{NumSentence, is_numeric},
    utils::str_is_chinese,
    zh::{
        ZhMode, ZhSentence,
        g2pw::{G2PW, G2PWOut},
    },
};

pub mod bert;
pub mod dict;
pub mod en;
pub mod num;
pub mod phone_symbol;
pub mod zh;

mod utils;

// Regex to filter out a wide range of emojis and symbols.
static CLEANUP_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{2600}-\u{27BF}]+",
    )
    .unwrap()
});

// Regex for splitting text into sentences based on strong terminators.
static SENTENCE_END_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"([。.?！!；;\n])").unwrap());

/// Filters out emojis and other non-essential symbols from the text.
fn cleanup_text(text: &str) -> String {
    CLEANUP_REGEX.replace_all(text, " ").into_owned()
}

fn merge_short_sentences(sentences: Vec<&str>) -> Vec<String> {
    let mut result = Vec::new();
    let mut i: usize = 0;

    while i < sentences.len() {
        let current = sentences[i];
        let word_count = if str_is_chinese(current) {
            current.len()
        } else {
            current.split_whitespace().count()
        };

        if word_count < 4 && i + 1 < sentences.len() {
            // Merge with next sentence
            let merged = format!("{} {}", current, sentences[i + 1]);
            result.push(merged);
            i += 2; // Skip next because it's merged
        } else {
            result.push((current).to_owned());
            i += 1;
        }
    }

    result
}

/// Splits text into chunks, handling long sentences gracefully.
///
/// This version is corrected to be compatible with older versions of the `regex` crate.
pub fn split_text(text: &str, max_len: usize) -> Vec<String> {
    debug!("Original text length: {}", text.chars().count());
    let mut chunks = Vec::new();

    // --- FIX START ---
    // Manually implement `split_inclusive` logic using `find_iter` for compatibility.
    let mut sentences_parts = Vec::new();
    let mut last_end = 0;
    // Dereference the Lazy wrapper to access the Regex object.
    for mat in SENTENCE_END_REGEX.find_iter(text) {
        sentences_parts.push(&text[last_end..mat.end()]);
        last_end = mat.end();
    }
    if last_end < text.len() {
        sentences_parts.push(&text[last_end..]);
    }
    // --- FIX END ---

    //
    let sentences_parts = merge_short_sentences(sentences_parts);

    for sentence in sentences_parts {
        let sentence_trimmed = sentence.trim();
        if sentence_trimmed.is_empty() {
            continue;
        }

        if sentence_trimmed.chars().count() <= max_len {
            chunks.push(sentence_trimmed.to_string());
            continue;
        }

        let mut current_pos = 0;
        while sentence_trimmed[current_pos..].chars().count() > max_len {
            let chunk_candidate = &sentence_trimmed[current_pos..];
            let mut window_end = 0;
            // Correctly calculate the end of the window based on char count
            for (i, c) in chunk_candidate.char_indices() {
                if window_end >= max_len {
                    break;
                }
                window_end = i + c.len_utf8();
            }
            let window = &chunk_candidate[..window_end];

            match window.rfind(|c: char| ",，:：".contains(c)) {
                Some(split_idx) => {
                    let split_point = current_pos + split_idx + 1; // Split after the punctuation
                    chunks.push(
                        sentence_trimmed[current_pos..split_point]
                            .trim()
                            .to_string(),
                    );
                    current_pos = split_point;
                }
                None => {
                    let split_point = current_pos + window_end;
                    chunks.push(
                        sentence_trimmed[current_pos..split_point]
                            .trim()
                            .to_string(),
                    );
                    current_pos = split_point;
                }
            }
        }

        let remaining_part = sentence_trimmed[current_pos..].trim();
        if !remaining_part.is_empty() {
            chunks.push(remaining_part.to_string());
        }
    }

    debug!("Split into {} chunks", chunks.len());
    chunks.into_iter().filter(|c| !c.is_empty()).collect()
}

#[derive(Debug, Clone, Copy)]
pub enum Lang {
    Zh,
    En,
}

#[derive(Debug, Clone, Copy)]
pub enum LangId {
    Auto,    // Mandarin
    AutoYue, // Cantonese
}

pub trait SentenceProcessor {
    fn get_text_for_bert(&self) -> String;
    fn get_word2ph(&self) -> &[i32];
}

impl SentenceProcessor for EnSentence {
    fn get_text_for_bert(&self) -> String {
        self.get_text_string()
    }

    fn get_word2ph(&self) -> &[i32] {
        &self.word2ph
    }
}

impl SentenceProcessor for ZhSentence {
    fn get_text_for_bert(&self) -> String {
        self.text.clone()
    }

    fn get_word2ph(&self) -> &[i32] {
        &self.word2ph
    }
}

pub struct TextProcessor {
    pub jieba: Arc<Jieba>,
    pub g2pw: G2PW,
    pub g2p_en: G2pEn,
    pub bert_model: BertModel,
}

impl TextProcessor {
    pub fn new(g2pw: G2PW, g2p_en: G2pEn, bert_model: BertModel) -> Result<Self> {
        Ok(Self {
            jieba: Arc::new(Jieba::new()),
            g2pw,
            g2p_en,
            bert_model,
        })
    }

    pub fn get_phone_and_bert(
        &mut self,
        text: &str,
        lang_id: LangId,
    ) -> Result<Vec<(String, Vec<i64>, Array2<f32>)>> {
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("Input text is empty"));
        }

        let cleaned_text = cleanup_text(text);
        let chunks = split_text(&cleaned_text, 64);
        let mut result = Vec::with_capacity(chunks.len().min(100));

        for chunk in chunks.iter() {
            debug!("Processing chunk: {}", chunk);
            let mut phone_builder = PhoneBuilder::new();
            phone_builder.extend_text(&self.jieba, chunk);

            if !chunk.ends_with(['。', '.', '?', '？', '!', '！', '；', ';', '\n']) {
                phone_builder.push_punctuation(".");
            }

            let mut chunk_phone_seq = Vec::with_capacity(128);
            let mut chunk_bert_features: Option<Array2<f32>> = None;

            for sentence in phone_builder.sentences {
                match self.process_sentence(sentence, lang_id) {
                    Ok((phone_seq, bert_features)) => {
                        if phone_seq.is_empty() {
                            continue;
                        }
                        chunk_phone_seq.extend(phone_seq);
                        chunk_bert_features = Some(match chunk_bert_features {
                            Some(existing) => concatenate![Axis(0), existing, bert_features],
                            None => bert_features,
                        });
                    }
                    Err(e) => {
                        warn!("Failed to process sentence in chunk '{}': {}", chunk, e);
                        if cfg!(debug_assertions) {
                            return Err(e);
                        }
                        continue;
                    }
                }
            }

            if !chunk_phone_seq.is_empty() && chunk_bert_features.is_some() {
                result.push((chunk.clone(), chunk_phone_seq, chunk_bert_features.unwrap()));
            }
        }

        if result.is_empty() {
            return Err(anyhow::anyhow!("No phonemes generated for text: {}", text));
        }
        Ok(result)
    }

    fn process_sentence(
        &mut self,
        sentence: Sentence,
        lang_id: LangId,
    ) -> Result<(Vec<i64>, Array2<f32>)> {
        let mut sentence = sentence;
        match &mut sentence {
            Sentence::Zh(zh) => {
                let mode = if matches!(lang_id, LangId::AutoYue) {
                    ZhMode::Cantonese
                } else {
                    ZhMode::Mandarin
                };
                zh.g2p(&mut self.g2pw, mode);
            }
            Sentence::En(en) => en
                .g2p(&mut self.g2p_en)
                .context("Failed to process English G2P")?,
            Sentence::Num(num) => {
                let mut phone_seq = Vec::with_capacity(64);
                let mut bert_features: Option<Array2<f32>> = None;
                for sub_sentence in num
                    .to_phone_sentence()
                    .context("Failed to convert number to phone sentence")?
                {
                    let (sub_phone_seq, sub_bert) = self.process_sentence(sub_sentence, lang_id)?;
                    phone_seq.extend(sub_phone_seq);
                    bert_features = Some(match bert_features {
                        Some(existing) => concatenate![Axis(0), existing, sub_bert],
                        None => sub_bert,
                    });
                }
                let seq_len = phone_seq.len();
                return Ok((
                    phone_seq,
                    bert_features.unwrap_or_else(|| Array2::zeros((seq_len, 1024))),
                ));
            }
        }

        let (phone_seq, text, word2ph) = match &sentence {
            Sentence::Zh(zh) => (
                zh.build_phone()
                    .context("Failed to build Chinese phonemes")?,
                zh.get_text_for_bert(),
                zh.get_word2ph(),
            ),
            Sentence::En(en) => (
                en.build_phone()
                    .context("Failed to build English phonemes")?,
                en.get_text_for_bert(),
                en.get_word2ph(),
            ),
            _ => unreachable!(),
        };

        let bert_features = self
            .bert_model
            .get_bert(&text, word2ph, phone_seq.len())
            .context("Failed to generate BERT features")?;
        Ok((phone_seq, bert_features))
    }
}

fn parse_punctuation(p: &str) -> Option<&'static str> {
    match p {
        "，" | "," => Some(","),
        "。" | "." => Some("."),
        "！" | "!" => Some("!"),
        "？" | "?" => Some("."),
        "；" | ";" => Some("."),
        "：" | ":" => Some(","),
        "‘" | "’" | "'" => Some("'"),
        "“" | "”" | "\"" => Some("-"),
        "（" | "(" => Some("-"),
        "）" | ")" => Some("-"),
        "【" | "[" => Some("-"),
        "】" | "]" => Some("-"),
        "《" | "<" => Some("-"),
        "》" | ">" => Some("-"),
        "—" => Some("-"),
        "～" | "~" | "…" | "_" | "..." => Some("…"),
        "·" => Some(","),
        "、" => Some(","),
        "$" => Some("."),
        "/" => Some(","),
        "\n" => Some("."),
        " " => Some(" "),
        _ => None,
    }
}

#[derive(Debug)]
enum Sentence {
    Zh(ZhSentence),
    En(EnSentence),
    Num(NumSentence),
}

struct PhoneBuilder {
    sentences: LinkedList<Sentence>,
}

impl PhoneBuilder {
    fn new() -> Self {
        Self {
            sentences: LinkedList::new(),
        }
    }

    fn extend_text(&mut self, jieba: &Jieba, text: &str) {
        for t in jieba.cut(text, true) {
            match parse_punctuation(t) {
                Some(p) => self.push_punctuation(p),
                None => {
                    if is_numeric(t) {
                        self.push_num_word(t);
                    } else if utils::str_is_chinese(t) {
                        self.push_zh_word(t);
                    } else if t.is_ascii() && !t.trim().is_empty() {
                        self.push_en_word(t);
                    } else {
                        info!("Skipping invalid word: {} in {}", t, text);
                    }
                }
            }
        }
    }

    fn push_punctuation(&mut self, p: &'static str) {
        match self.sentences.back_mut() {
            Some(Sentence::Zh(zh)) => {
                zh.text.push_str(if p == " " { "," } else { p });
                zh.phones.push(G2PWOut::RawChar(p.chars().next().unwrap()));
            }
            Some(Sentence::En(en)) => {
                if p == " "
                    && en
                        .text
                        .last()
                        .map(|w| matches!(w, EnWord::Word(w) if w == "a"))
                        .unwrap_or(false)
                {
                    return;
                }
                en.text.push(EnWord::Punctuation(p));
            }
            Some(Sentence::Num(n)) => {
                if n.need_drop() {
                    self.sentences.pop_back();
                }
                self.sentences.push_back(Sentence::En(EnSentence {
                    phone_ids: Vec::with_capacity(16),
                    phones: Vec::with_capacity(16),
                    text: vec![EnWord::Punctuation(p)],
                    word2ph: Vec::with_capacity(16),
                }));
            }
            _ => {
                debug!("Skipping punctuation: {}", p);
            }
        }
    }

    fn push_en_word(&mut self, word: &str) {
        let word = word.to_ascii_lowercase();
        match self.sentences.back_mut() {
            Some(Sentence::En(en)) => {
                if let Some(EnWord::Punctuation(p)) = en.text.last() {
                    if *p == "'" || *p == "-" {
                        let p = en.text.pop().unwrap();
                        if let Some(EnWord::Word(last_word)) = en.text.last_mut() {
                            if let EnWord::Punctuation(p_str) = p {
                                last_word.push_str(p_str);
                                last_word.push_str(&word);
                                return;
                            }
                        }
                        en.text.push(p);
                    }
                }
                en.text.push(EnWord::Word(word));
            }
            Some(Sentence::Num(n)) if n.need_drop() => {
                let pop = self.sentences.pop_back().unwrap();
                if let Sentence::Num(n) = pop {
                    if n.is_link_symbol() {
                        self.push_punctuation("-");
                    }
                }
                self.push_en_word(&word);
            }
            _ => {
                let en = EnSentence {
                    phone_ids: Vec::with_capacity(16),
                    phones: Vec::with_capacity(16),
                    text: vec![EnWord::Word(word)],
                    word2ph: Vec::with_capacity(16),
                };
                self.sentences.push_back(Sentence::En(en));
            }
        }
    }

    fn push_zh_word(&mut self, word: &str) {
        fn add_zh_word(zh: &mut ZhSentence, word: &str) {
            zh.text.push_str(word);
            match dict::zh_word_dict(word) {
                Some(phones) => {
                    zh.phones.extend(
                        phones
                            .into_iter()
                            .map(|arg0: &std::string::String| G2PWOut::Pinyin(arg0.clone())),
                    );
                }
                None => {
                    zh.phones
                        .extend(word.chars().map(|_| G2PWOut::Pinyin(String::new())));
                }
            }
        }

        match self.sentences.back_mut() {
            Some(Sentence::Zh(zh)) => add_zh_word(zh, word),
            Some(Sentence::Num(n)) if n.need_drop() => {
                self.sentences.pop_back();
                self.push_zh_word(word);
            }
            _ => {
                let mut zh = ZhSentence {
                    phone_ids: Vec::with_capacity(16),
                    phones: Vec::with_capacity(16),
                    word2ph: Vec::with_capacity(16),
                    text: String::with_capacity(32),
                };
                add_zh_word(&mut zh, word);
                self.sentences.push_back(Sentence::Zh(zh));
            }
        }
    }

    fn push_num_word(&mut self, word: &str) {
        let lang = match self.sentences.back() {
            Some(Sentence::En(_)) => Lang::En,
            Some(Sentence::Num(n)) => n.lang,
            _ => Lang::Zh,
        };

        match self.sentences.back_mut() {
            Some(Sentence::Num(num)) => {
                num.text.push_str(word);
            }
            _ => {
                self.sentences.push_back(Sentence::Num(NumSentence {
                    text: word.to_string(),
                    lang,
                }));
            }
        }
    }
}
