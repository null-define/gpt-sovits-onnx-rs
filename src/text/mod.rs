// text/mod.rs
use std::{collections::LinkedList, fmt::Debug};

use anyhow::Result;
use jieba_rs::Jieba;
use log::{debug, info, warn};
use ndarray::{Array2, Axis, concatenate};
use regex::Regex;

use crate::text::{
    bert::BertModel,
    en::{EnSentence, EnWord, g2p_en::G2pEn},
    num::{NumSentence, is_numeric},
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

#[derive(Debug, Clone, Copy)]
pub enum Lang {
    Zh,
    En,
}

#[derive(Debug, Clone, Copy)]
pub enum LangId {
    Auto,
    AutoYue,
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
    pub jieba: Jieba,
    pub g2pw: G2PW,
    pub g2p_en: G2pEn,
    pub bert_model: BertModel,
}

impl TextProcessor {
    /// Creates a new TextProcessor with initialized Jieba, G2PW, and BERT model.
    pub fn new(g2pw: G2PW, g2p_en: G2pEn, bert_model: BertModel) -> Result<Self> {
        Ok(Self {
            jieba: Jieba::new(),
            g2pw,
            g2p_en,
            bert_model,
        })
    }

    /// Processes text into chunks, generating phoneme IDs and BERT features for each chunk.
    pub fn get_phone_and_bert(
        &mut self,
        text: &str,
        lang_id: LangId,
    ) -> Result<Vec<(String, Vec<i64>, Array2<f32>)>> {
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("Input text is empty"));
        }

        let chunks = split_text(text, usize::MAX);
        let mut result = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            debug!("Processing chunk: {:?}", chunk);
            let mut phone_builder = PhoneBuilder::new();
            phone_builder.extend_text(&self.jieba, &chunk);

            if !chunk.ends_with(['。', '.', '?', '？', '!', '！']) {
                phone_builder.push_punctuation(".");
            }

            let mut chunk_phone_seq = Vec::new();
            let mut chunk_bert_features: Option<Array2<f32>> = None;

            for sentence in phone_builder.sentences {
                let (phone_seq, bert_features) = match self.process_sentence(sentence, lang_id) {
                    Ok(res) => res,
                    Err(e) => {
                        warn!("Failed to process sentence in chunk '{}': {}", chunk, e);
                        if cfg!(debug_assertions) {
                            return Err(e);
                        }
                        continue;
                    }
                };

                if phone_seq.is_empty() {
                    continue;
                }

                chunk_phone_seq.extend(phone_seq);
                chunk_bert_features = Some(match chunk_bert_features {
                    Some(existing) => concatenate![Axis(0), existing, bert_features],
                    None => bert_features,
                });
            }

            if !chunk_phone_seq.is_empty() && chunk_bert_features.is_some() {
                result.push((chunk, chunk_phone_seq, chunk_bert_features.unwrap()));
            } else {
                warn!(
                    "No phonemes or BERT features generated for chunk '{}'",
                    chunk
                );
            }
        }

        if result.is_empty() {
            return Err(anyhow::anyhow!("No phonemes generated for text: {}", text));
        }
        Ok(result)
    }

    /// Processes a single sentence, generating phoneme IDs and BERT features.
    fn process_sentence(
        &mut self,
        mut sentence: Sentence,
        lang_id: LangId,
    ) -> Result<(Vec<i64>, Array2<f32>)> {
        match &mut sentence {
            Sentence::Zh(zh) => {
                let mode = if matches!(lang_id, LangId::AutoYue) {
                    ZhMode::Cantonese
                } else {
                    ZhMode::Mandarin
                };
                zh.g2p(&mut self.g2pw, mode);
            }
            Sentence::En(en) => en.g2p(&mut self.g2p_en)?,
            Sentence::Num(num) => {
                let mut phone_seq = Vec::new();
                let mut bert_features: Option<Array2<f32>> = None;
                for sub_sentence in num.to_phone_sentence()? {
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
            Sentence::Zh(zh) => (zh.build_phone()?, zh.get_text_for_bert(), zh.get_word2ph()),
            Sentence::En(en) => (en.build_phone()?, en.get_text_for_bert(), en.get_word2ph()),
            _ => unreachable!(),
        };

        let bert_features = self.bert_model.get_bert(&text, word2ph, phone_seq.len())?;
        Ok((phone_seq, bert_features))
    }
}

/// Splits text into chunks based on punctuation.
pub fn split_text(text: &str, _max_chunk_size: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    let re = Regex::new(r"[。.?！!；;\n]").unwrap();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for segment in re.split(text) {
        current_chunk.push_str(segment);
        if re.is_match(segment) {
            let trimmed = current_chunk.trim();
            if !trimmed.is_empty() {
                chunks.push(trimmed.to_string());
            }
            current_chunk.clear();
        }
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }
    debug!("chunks {:?}", chunks);
    chunks
}

fn is_punctuation(c: char) -> bool {
    ['。', '.', '?', '？', '!', '！', ';', '；', '\n'].contains(&c)
}

fn parse_punctuation(p: &str) -> Option<&'static str> {
    match p {
        "，" | "," => Some(","),
        "。" | "." => Some("."),
        "！" | "!" => Some("!"),
        "？" | "?" => Some("."),
        "；" | ";" => Some("."),
        "：" | ":" => Some(","),
        "‘" | "’" => Some("'"),
        "'" => Some("'"),
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
                None if is_numeric(t) => self.push_num_word(t),
                None if utils::str_is_chinese(t) => self.push_zh_word(t),
                None if t.is_ascii() && !t.trim().is_empty() => self.push_en_word(t),
                _ => info!("skip word: {:?} in {}", t, text),
            }
        }
    }

    pub fn push_punctuation(&mut self, p: &'static str) {
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
                    phone_ids: vec![],
                    phones: vec![],
                    text: vec![EnWord::Punctuation(p)],
                    word2ph: vec![],
                }));
            }
            _ => {
                debug!("skip punctuation: {}", p);
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
                    phone_ids: vec![],
                    phones: vec![],
                    text: vec![EnWord::Word(word)],
                    word2ph: vec![],
                };
                self.sentences.push_back(Sentence::En(en));
            }
        }
    }

    fn push_zh_word(&mut self, word: &str) {
        fn h(zh: &mut ZhSentence, word: &str) {
            zh.text.push_str(word);
            match dict::zh_word_dict(word) {
                Some(phones) => {
                    for p in phones {
                        zh.phones.push(G2PWOut::Pinyin(p.clone()));
                    }
                }
                None => {
                    for _ in word.chars() {
                        zh.phones.push(G2PWOut::Pinyin(String::new()));
                    }
                }
            }
        }

        match self.sentences.back_mut() {
            Some(Sentence::Zh(zh)) => h(zh, word),
            Some(Sentence::Num(n)) if n.need_drop() => {
                self.sentences.pop_back();
                self.push_zh_word(word);
            }
            _ => {
                let mut zh = ZhSentence {
                    phone_ids: Vec::new(),
                    phones: Vec::new(),
                    word2ph: Vec::new(),
                    text: String::new(),
                };
                h(&mut zh, word);
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
