use std::{
    borrow::Cow,
    collections::{HashMap, LinkedList},
    fmt::Debug,
};

use anyhow::Result;
use arpabet::Arpabet;
use jieba_rs::Jieba;
use log::{debug, info, warn};
use ndarray::{Array2, Axis, concatenate};
use pest::Parser;

use crate::text::{
    bert::BertModel,
    g2pw::{G2PWConverter, G2PWOut},
};

pub mod bert;
pub mod dict;
pub mod g2pw;
pub mod num;
pub mod symbols;

mod utils;

// Static list of consonants for split_zh_ph_
static CONSONANTS: &[char] = &[
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y',
    'w',
];

#[inline]
fn get_phone_symbol(symbols: &HashMap<String, i64>, ph: &str) -> i64 {
    symbols.get(ph).copied().unwrap_or(3)
}

// Optimized phoneme splitting function for Chinese
// NOTE: This large match statement could be further optimized using a HashMap or a phf::Map
// if performance profiling indicates it as a bottleneck.
fn split_zh_ph(ph: &str) -> (&str, &str) {
    match ph {
        "a" => ("AA", "a5"),
        "a1" => ("AA", "a1"),
        "a2" => ("AA", "a2"),
        "a3" => ("AA", "a3"),
        "a4" => ("AA", "a4"),
        "a5" => ("AA", "a5"),
        "ai" => ("AA", "ai5"),
        "ai1" => ("AA", "ai1"),
        "ai2" => ("AA", "ai2"),
        "ai3" => ("AA", "ai3"),
        "ai4" => ("AA", "ai4"),
        "ai5" => ("AA", "ai5"),
        "an" => ("AA", "an5"),
        "an1" => ("AA", "an1"),
        "an2" => ("AA", "an2"),
        "an3" => ("AA", "an3"),
        "an4" => ("AA", "an4"),
        "an5" => ("AA", "an5"),
        "ang" => ("AA", "ang5"),
        "ang1" => ("AA", "ang1"),
        "ang2" => ("AA", "ang2"),
        "ang3" => ("AA", "ang3"),
        "ang4" => ("AA", "ang4"),
        "ang5" => ("AA", "ang5"),
        "ao" => ("AA", "ao5"),
        "ao1" => ("AA", "ao1"),
        "ao2" => ("AA", "ao2"),
        "ao3" => ("AA", "ao3"),
        "ao4" => ("AA", "ao4"),
        "ao5" => ("AA", "ao5"),
        "chi" => ("ch", "ir5"),
        "chi1" => ("ch", "ir1"),
        "chi2" => ("ch", "ir2"),
        "chi3" => ("ch", "ir3"),
        "chi4" => ("ch", "ir4"),
        "chi5" => ("ch", "ir5"),
        "ci" => ("c", "i05"),
        "ci1" => ("c", "i01"),
        "ci2" => ("c", "i02"),
        "ci3" => ("c", "i03"),
        "ci4" => ("c", "i04"),
        "ci5" => ("c", "i05"),
        "e" => ("EE", "e5"),
        "e1" => ("EE", "e1"),
        "e2" => ("EE", "e2"),
        "e3" => ("EE", "e3"),
        "e4" => ("EE", "e4"),
        "e5" => ("EE", "e5"),
        "ei" => ("EE", "ei5"),
        "ei1" => ("EE", "ei1"),
        "ei2" => ("EE", "ei2"),
        "ei3" => ("EE", "ei3"),
        "ei4" => ("EE", "ei4"),
        "ei5" => ("EE", "ei5"),
        "en" => ("EE", "en5"),
        "en1" => ("EE", "en1"),
        "en2" => ("EE", "en2"),
        "en3" => ("EE", "en3"),
        "en4" => ("EE", "en4"),
        "en5" => ("EE", "en5"),
        "eng" => ("EE", "eng5"),
        "eng1" => ("EE", "eng1"),
        "eng2" => ("EE", "eng2"),
        "eng3" => ("EE", "eng3"),
        "eng4" => ("EE", "eng4"),
        "eng5" => ("EE", "eng5"),
        "er" => ("EE", "er5"),
        "er1" => ("EE", "er1"),
        "er2" => ("EE", "er2"),
        "er3" => ("EE", "er3"),
        "er4" => ("EE", "er4"),
        "er5" => ("EE", "er5"),
        "ju" => ("j", "v5"),
        "ju1" => ("j", "v1"),
        "ju2" => ("j", "v2"),
        "ju3" => ("j", "v3"),
        "ju4" => ("j", "v4"),
        "ju5" => ("j", "v5"),
        "juan" => ("j", "van5"),
        "juan1" => ("j", "van1"),
        "juan2" => ("j", "van2"),
        "juan3" => ("j", "van3"),
        "juan4" => ("j", "van4"),
        "juan5" => ("j", "van5"),
        "jue" => ("j", "ve5"),
        "jue1" => ("j", "ve1"),
        "jue2" => ("j", "ve2"),
        "jue3" => ("j", "ve3"),
        "jue4" => ("j", "ve4"),
        "jue5" => ("j", "ve5"),
        "jun" => ("j", "vn5"),
        "jun1" => ("j", "vn1"),
        "jun2" => ("j", "vn2"),
        "jun3" => ("j", "vn3"),
        "jun4" => ("j", "vn4"),
        "jun5" => ("j", "vn5"),
        "o" => ("OO", "o5"),
        "o1" => ("OO", "o1"),
        "o2" => ("OO", "o2"),
        "o3" => ("OO", "o3"),
        "o4" => ("OO", "o4"),
        "o5" => ("OO", "o5"),
        "ou" => ("OO", "ou5"),
        "ou1" => ("OO", "ou1"),
        "ou2" => ("OO", "ou2"),
        "ou3" => ("OO", "ou3"),
        "ou4" => ("OO", "ou4"),
        "ou5" => ("OO", "ou5"),
        "qu" => ("q", "v5"),
        "qu1" => ("q", "v1"),
        "qu2" => ("q", "v2"),
        "qu3" => ("q", "v3"),
        "qu4" => ("q", "v4"),
        "qu5" => ("q", "v5"),
        "quan" => ("q", "van5"),
        "quan1" => ("q", "van1"),
        "quan2" => ("q", "van2"),
        "quan3" => ("q", "van3"),
        "quan4" => ("q", "van4"),
        "quan5" => ("q", "van5"),
        "que" => ("q", "ve5"),
        "que1" => ("q", "ve1"),
        "que2" => ("q", "ve2"),
        "que3" => ("q", "ve3"),
        "que4" => ("q", "ve4"),
        "que5" => ("q", "ve5"),
        "qun" => ("q", "vn5"),
        "qun1" => ("q", "vn1"),
        "qun2" => ("q", "vn2"),
        "qun3" => ("q", "vn3"),
        "qun4" => ("q", "vn4"),
        "qun5" => ("q", "vn5"),
        "ri" => ("r", "ir5"),
        "ri1" => ("r", "ir1"),
        "ri2" => ("r", "ir2"),
        "ri3" => ("r", "ir3"),
        "ri4" => ("r", "ir4"),
        "ri5" => ("r", "ir5"),
        "xu" => ("x", "v5"),
        "xu1" => ("x", "v1"),
        "xu2" => ("x", "v2"),
        "xu3" => ("x", "v3"),
        "xu4" => ("x", "v4"),
        "xu5" => ("x", "v5"),
        "xuan" => ("x", "van5"),
        "xuan1" => ("x", "van1"),
        "xuan2" => ("x", "van2"),
        "xuan3" => ("x", "van3"),
        "xuan4" => ("x", "van4"),
        "xuan5" => ("x", "van5"),
        "xue" => ("x", "ve5"),
        "xue1" => ("x", "ve1"),
        "xue2" => ("x", "ve2"),
        "xue3" => ("x", "ve3"),
        "xue4" => ("x", "ve4"),
        "xue5" => ("x", "ve5"),
        "xun" => ("x", "vn5"),
        "xun1" => ("x", "vn1"),
        "xun2" => ("x", "vn2"),
        "xun3" => ("x", "vn3"),
        "xun4" => ("x", "vn4"),
        "xun5" => ("x", "vn5"),
        "yan" => ("y", "En5"),
        "yan1" => ("y", "En1"),
        "yan2" => ("y", "En2"),
        "yan3" => ("y", "En3"),
        "yan4" => ("y", "En4"),
        "yan5" => ("y", "En5"),
        "ye" => ("y", "E5"),
        "ye1" => ("y", "E1"),
        "ye2" => ("y", "E2"),
        "ye3" => ("y", "E3"),
        "ye4" => ("y", "E4"),
        "ye5" => ("y", "E5"),
        "yu" => ("y", "v5"),
        "yu1" => ("y", "v1"),
        "yu2" => ("y", "v2"),
        "yu3" => ("y", "v3"),
        "yu4" => ("y", "v4"),
        "yu5" => ("y", "v5"),
        "yuan" => ("y", "van5"),
        "yuan1" => ("y", "van1"),
        "yuan2" => ("y", "van2"),
        "yuan3" => ("y", "van3"),
        "yuan4" => ("y", "van4"),
        "yuan5" => ("y", "van5"),
        "yue" => ("y", "ve5"),
        "yue1" => ("y", "ve1"),
        "yue2" => ("y", "ve2"),
        "yue3" => ("y", "ve3"),
        "yue4" => ("y", "ve4"),
        "yue5" => ("y", "ve5"),
        "yun" => ("y", "vn5"),
        "yun1" => ("y", "vn1"),
        "yun2" => ("y", "vn2"),
        "yun3" => ("y", "vn3"),
        "yun4" => ("y", "vn4"),
        "yun5" => ("y", "vn5"),
        "zhi" => ("zh", "ir5"),
        "zhi1" => ("zh", "ir1"),
        "zhi2" => ("zh", "ir2"),
        "zhi3" => ("zh", "ir3"),
        "zhi4" => ("zh", "ir4"),
        "zhi5" => ("zh", "ir5"),
        "zi" => ("z", "i05"),
        "zi1" => ("z", "i01"),
        "zi2" => ("z", "i02"),
        "zi3" => ("z", "i03"),
        "zi4" => ("z", "i04"),
        "zi5" => ("z", "i05"),
        "shi" => ("sh", "ir5"),
        "shi1" => ("sh", "ir1"),
        "shi2" => ("sh", "ir2"),
        "shi3" => ("sh", "ir3"),
        "shi4" => ("sh", "ir4"),
        "shi5" => ("sh", "ir5"),
        "si" => ("s", "i05"),
        "si1" => ("s", "i01"),
        "si2" => ("s", "i02"),
        "si3" => ("s", "i03"),
        "si4" => ("s", "i04"),
        "si5" => ("s", "i05"),
        ph => match split_zh_ph_(ph) {
            (y, "ü") => (y, "v5"),
            (y, "ü1") => (y, "v1"),
            (y, "ü2") => (y, "v2"),
            (y, "ü3") => (y, "v3"),
            (y, "ü4") => (y, "v4"),
            (y, "üe") => (y, "ve5"),
            (y, "üe1") => (y, "ve1"),
            (y, "üe2") => (y, "ve2"),
            (y, "üe3") => (y, "ve3"),
            (y, "üe4") => (y, "ve4"),
            (y, "üan") => (y, "van5"),
            (y, "üan1") => (y, "van1"),
            (y, "üan2") => (y, "van2"),
            (y, "üan3") => (y, "van3"),
            (y, "üan4") => (y, "van4"),
            (y, "ün") => (y, "vn5"),
            (y, "ün1") => (y, "vn1"),
            (y, "ün2") => (y, "vn2"),
            (y, "ün3") => (y, "vn3"),
            (y, "ün4") => (y, "vn4"),
            (y, "a") => (y, "a5"),
            (y, "o") => (y, "o5"),
            (y, "e") => (y, "e5"),
            (y, "i") => (y, "i5"),
            (y, "u") => (y, "u5"),
            (y, "ai") => (y, "ai5"),
            (y, "ei") => (y, "ei5"),
            (y, "ao") => (y, "ao5"),
            (y, "ou") => (y, "ou5"),
            (y, "ia") => (y, "ia5"),
            (y, "ie") => (y, "ie5"),
            (y, "ua") => (y, "ua5"),
            (y, "uo") => (y, "uo5"),
            (y, "iao") => (y, "iao5"),
            (y, "iou") => (y, "iou5"),
            (y, "uai") => (y, "uai5"),
            (y, "uei") => (y, "uei5"),
            (y, "an") => (y, "an5"),
            (y, "en") => (y, "en5"),
            (y, "ang") => (y, "ang5"),
            (y, "eng") => (y, "eng5"),
            (y, "ian") => (y, "ian5"),
            (y, "in") => (y, "in5"),
            (y, "iang") => (y, "iang5"),
            (y, "ing") => (y, "ing5"),
            (y, "uan") => (y, "uan5"),
            (y, "un") => (y, "un5"),
            (y, "uang") => (y, "uang5"),
            (y, "ong") => (y, "ong5"),
            (y, "er") => (y, "er5"),
            (y, s) => (y, s),
        },
    }
}

// Helper function to split phonemes based on consonants
#[inline]
fn split_zh_ph_(ph: &str) -> (&str, &str) {
    if ph.starts_with("zh") || ph.starts_with("ch") || ph.starts_with("sh") {
        ph.split_at(2)
    } else if ph.chars().next().map_or(false, |c| CONSONANTS.contains(&c)) {
        ph.split_at(1)
    } else {
        (ph, ph)
    }
}

pub struct TextProcessor {
    pub jieba: Jieba,
    pub g2pw: G2PWConverter,
    pub bert_model: BertModel,
    pub symbols: HashMap<String, i64>,
    // REFACTOR: Added Arpabet dictionary to the processor struct.
    // This avoids reloading the dictionary on every call, improving performance.
    pub arpabet: Arpabet,
}

impl TextProcessor {
    // REFACTOR: Added a constructor to initialize the processor, including the Arpabet dictionary.
    pub fn new(
        jieba: Jieba,
        g2pw: G2PWConverter,
        bert_model: BertModel,
        symbols: HashMap<String, i64>,
    ) -> Result<Self> {
        info!("Loading Arpabet CMU dict...");
        let arpabet = arpabet::load_cmudict();
        info!("Arpabet CMU dict loaded.");
        Ok(Self {
            jieba,
            g2pw,
            bert_model,
            symbols,
            arpabet: arpabet.clone(),
        })
    }

    /// REFACTOR: Main function rewritten to process sentences individually for BERT generation.
    /// This separates the logic for Chinese and English, as requested.
    pub fn get_phone_and_bert(
        &mut self,
        text: &str,
    ) -> Result<Vec<(String, Vec<i64>, Array2<f32>)>> {
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("Input text is empty"));
        }

        let chunks = split_text(text, usize::MAX);
        let mut result = Vec::new();

        for chunk in &chunks {
            debug!("Processing chunk: {:?}", chunk);
            let mut phone_builder = PhoneBuilder::new();
            phone_builder.extend_text(&self.jieba, chunk);

            // Ensure chunk ends with punctuation for consistent processing
            if !chunk.ends_with(['。', '.', '?', '？', '!', '！']) {
                phone_builder.push_punctuation(".");
            }

            let mut chunk_phone_seq = Vec::new();
            let mut chunk_bert_features: Option<Array2<f32>> = None;

            // Process each sentence (Zh, En, Num) in the chunk individually
            for sentence in phone_builder.sentences {
                let (sentence_phone_seq, sentence_bert) = match self.process_sentence(sentence) {
                    Ok(res) => res,
                    Err(e) => {
                        warn!("Failed to process a sentence: {}", e);
                        if cfg!(debug_assertions) {
                            return Err(e);
                        }
                        continue;
                    }
                };

                // Skip if sentence processing yielded no phonemes
                if sentence_phone_seq.is_empty() {
                    continue;
                }

                // Append the results for this sentence to the chunk's aggregates
                chunk_phone_seq.extend(sentence_phone_seq);

                if let Some(existing_bert) = chunk_bert_features {
                    chunk_bert_features = Some(concatenate![Axis(0), existing_bert, sentence_bert]);
                } else {
                    chunk_bert_features = Some(sentence_bert);
                }
            }

            if !chunk_phone_seq.is_empty() {
                if let Some(bert) = chunk_bert_features {
                    result.push((chunk.clone(), chunk_phone_seq, bert));
                } else {
                    warn!(
                        "Phonemes generated for chunk '{}', but no BERT features.",
                        chunk
                    );
                }
            }
        }

        if result.is_empty() {
            return Err(anyhow::anyhow!("No phonemes generated for text: {}", text));
        }
        Ok(result)
    }

    /// REFACTOR: Helper function to process a single Sentence enum.
    /// It encapsulates the logic for handling Zh, En, and Num sentence types,
    /// generating phonemes and BERT features for each.
    fn process_sentence(&mut self, sentence: Sentence) -> Result<(Vec<i64>, Array2<f32>)> {
        match sentence {
            Sentence::Zh(mut zh) => {
                debug!("Processing Zh text: {:?}", zh.text);
                zh.generate_pinyin(self);
                let phone_seq = zh.build_phone()?;
                // if phone_seq.is_empty() {
                //     return Ok((vec![], Array2::zeros((0, self.bert_model.hidden_size()))));
                // }
                let bert_features =
                    self.bert_model
                        .get_bert(&zh.text, &zh.word2ph, phone_seq.len())?;
                Ok((phone_seq, bert_features))
            }
            Sentence::En(mut en) => {
                debug!("Processing En text: {:?}", en.get_text_string());
                en.generate_phones(self);
                let phone_seq = en.build_phone()?;
                // if phone_seq.is_empty() {
                //     return Ok((vec![], Array2::zeros((0, self.bert_model.hidden_size()))));
                // }
                let en_text_for_bert = en.get_text_string();
                let bert_features =
                    self.bert_model
                        .get_bert(&en_text_for_bert, &en.word2ph, phone_seq.len())?;
                Ok((phone_seq, bert_features))
            }
            Sentence::Num(num) => {
                debug!("Processing Num text: {:?}", num.text);
                let mut num_phone_seq = Vec::new();
                let mut num_bert_features: Option<Array2<f32>> = None;

                // Numbers are converted into a sequence of Zh or En sentences, which we process here.
                for sub_sentence in num.to_phone_sentence()? {
                    // Recurse to process the sub-sentence.
                    // This is safe as Num -> Zh/En, so no infinite recursion.
                    let (sub_phone_seq, sub_bert) = self.process_sentence(sub_sentence)?;

                    num_phone_seq.extend(sub_phone_seq);
                    if let Some(existing_bert) = num_bert_features {
                        num_bert_features = Some(concatenate![Axis(0), existing_bert, sub_bert]);
                    } else {
                        num_bert_features = Some(sub_bert);
                    }
                }

                let seq_len = num_phone_seq.len();
                Ok((
                    num_phone_seq,
                    num_bert_features.unwrap_or_else(|| Array2::zeros((seq_len, 1024))),
                ))
            }
        }
    }
}

pub fn split_text(text: &str, _max_chunk_size: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    let is_en = text.is_ascii();
    let mut chunks = Vec::new();
    let mut current_chunk = String::with_capacity(text.len());
    let mut total_count = 0;

    // Split text by punctuation to form chunks.
    for segment in text.split_inclusive(is_punctuation) {
        let count = if is_en {
            segment.split_whitespace().count()
        } else {
            segment.chars().count()
        };

        debug!(
            "segment: {:?}, count: {}, total_count: {}",
            segment, count, total_count
        );

        current_chunk.push_str(segment);
        total_count += count;

        // Heuristic to create a new chunk
        if total_count > 4 && (segment.ends_with(['。', '.', '?', '？', '!', '！', '\n'])) {
            let trimmed = current_chunk.trim();
            if !trimmed.is_empty() {
                chunks.push(trimmed.to_string());
            }
            current_chunk.clear();
            total_count = 0;
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

#[derive(Debug)]
struct ZhSentence {
    phone_ids: Vec<i64>,
    phones: Vec<G2PWOut>,
    word2ph: Vec<i32>,
    text: String,
}

impl ZhSentence {
    fn generate_pinyin(&mut self, processor: &mut TextProcessor) {
        // Attempt to get pinyin using the primary G2P model
        let pinyin = processor.g2pw.get_pinyin(&self.text).unwrap_or_else(|e| {
            warn!("Pinyin generation failed: {}. Using fallback.", e);
            processor.g2pw.simple_get_pinyin(&self.text)
        });

        // Ensure phoneme list is synchronized with pinyin results
        if pinyin.len() != self.phones.len() {
            warn!(
                "Pinyin length mismatch: {} (pinyin) vs {} (phones) for text '{}'",
                pinyin.len(),
                self.phones.len(),
                self.text
            );
            // In case of mismatch, trust the new pinyin result
            self.phones = pinyin;
        } else {
            // Fill in missing phonemes
            for (i, out) in pinyin.into_iter().enumerate() {
                if matches!(self.phones[i], G2PWOut::Pinyin("") | G2PWOut::RawChar(_)) {
                    self.phones[i] = out;
                }
            }
        }
        debug!("phones: {:?}", self.phones);

        // Convert pinyin to phone IDs and generate the word-to-phoneme map
        for p in &self.phones {
            match p {
                G2PWOut::Pinyin(p) => {
                    let (initial, final_) = split_zh_ph(p);
                    self.phone_ids
                        .push(get_phone_symbol(&processor.symbols, initial));
                    if !final_.is_empty() {
                        self.phone_ids
                            .push(get_phone_symbol(&processor.symbols, final_));
                        self.word2ph.push(2);
                    } else {
                        self.word2ph.push(1);
                    }
                }
                G2PWOut::RawChar(c) => {
                    self.phone_ids
                        .push(get_phone_symbol(&processor.symbols, &c.to_string()));
                    self.word2ph.push(1);
                }
            }
        }
    }

    fn build_phone(&self) -> Result<Vec<i64>> {
        Ok(self.phone_ids.clone())
    }
}

#[derive(PartialEq, Eq)]
enum EnWord {
    Word(String),
    Punctuation(&'static str),
}

impl std::fmt::Debug for EnWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnWord::Word(w) => write!(f, "\"{}\"", w),
            EnWord::Punctuation(p) => write!(f, "\"{}\"", p),
        }
    }
}

#[derive(Debug)]
struct EnSentence {
    phone_ids: Vec<i64>,
    phones: Vec<Cow<'static, str>>,
    text: Vec<EnWord>,
    // REFACTOR: Added word2ph to EnSentence to correctly map words to their phoneme counts.
    word2ph: Vec<i32>,
}

impl EnSentence {
    // REFACTOR: `generate_phones` now uses the pre-loaded dictionary from the `TextProcessor`.
    // It also correctly calculates the `word2ph` mapping.
    fn generate_phones(&mut self, processor: &TextProcessor) {
        for word in &self.text {
            let mut ph_count_for_word = 0;
            match word {
                EnWord::Word(w) => {
                    // Use the arpabet dictionary from the processor
                    if let Some(phones) = processor.arpabet.get_polyphone_str(w) {
                        for ph in phones {
                            self.phones.push(Cow::Borrowed(ph));
                            self.phone_ids
                                .push(get_phone_symbol(&processor.symbols, ph));
                            ph_count_for_word += 1;
                        }
                    } else {
                        // Fallback for out-of-vocabulary words: process character by character
                        warn!(
                            "Word '{}' not in CMUdict, falling back to char processing.",
                            w
                        );
                        for c in w.chars() {
                            let c_str = c.to_string();
                            if let Some(phones) = processor.arpabet.get_polyphone_str(&c_str) {
                                for ph in phones {
                                    self.phones.push(Cow::Borrowed(ph));
                                    self.phone_ids
                                        .push(get_phone_symbol(&processor.symbols, ph));
                                    ph_count_for_word += 1;
                                }
                            } else {
                                self.phones.push(Cow::Owned(c_str.clone()));
                                self.phone_ids
                                    .push(get_phone_symbol(&processor.symbols, &c_str));
                                ph_count_for_word += 1;
                            }
                        }
                    }
                }
                EnWord::Punctuation(p) => {
                    self.phones.push(Cow::Borrowed(p));
                    self.phone_ids.push(get_phone_symbol(&processor.symbols, p));
                    ph_count_for_word += 1;
                }
            }
            // Only push to word2ph if phonemes were generated for this word/punctuation.
            if ph_count_for_word > 0 {
                self.word2ph.push(ph_count_for_word);
            }
        }
        debug!("EnSentence phones: {:?}", self.phones);
        debug!("EnSentence phone_ids: {:?}", self.phone_ids);
        debug!("EnSentence word2ph: {:?}", self.word2ph);
    }

    fn build_phone(&self) -> Result<Vec<i64>> {
        Ok(self.phone_ids.clone())
    }

    /// Helper to reconstruct the original text from the EnWord vector for BERT processing.
    fn get_text_string(&self) -> String {
        self.text
            .iter()
            .map(|w| match w {
                EnWord::Word(s) => s.as_str(),
                EnWord::Punctuation(p) => p,
            })
            .collect::<String>()
    }
}

#[derive(Debug, Clone, Copy)]
enum Lang {
    Zh,
    En,
}

#[derive(Debug)]
struct NumSentence {
    text: String,
    lang: Lang,
}

static NUM_OP: [char; 8] = ['+', '-', '*', '×', '/', '÷', '=', '%'];

impl NumSentence {
    fn need_drop(&self) -> bool {
        let num_text = self.text.trim();
        num_text.is_empty() || num_text.chars().all(|c| NUM_OP.contains(&c))
    }

    fn is_link_symbol(&self) -> bool {
        self.text == "-"
    }

    fn to_phone_sentence(&self) -> Result<LinkedList<Sentence>> {
        let mut builder = PhoneBuilder::new();
        let pairs = num::ExprParser::parse(num::Rule::all, &self.text)?;
        for pair in pairs {
            match self.lang {
                Lang::Zh => num::zh::parse_all(pair, &mut builder)?,
                Lang::En => num::en::parse_all(pair, &mut builder)?,
            }
        }
        Ok(builder.sentences)
    }
}

#[derive(Debug)]
enum Sentence {
    Zh(ZhSentence),
    En(EnSentence),
    Num(NumSentence),
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


fn is_numeric(p: &str) -> bool {
    p.chars().any(|c| c.is_numeric())
        || p.contains(&NUM_OP)
        || p.to_lowercase().contains(&[
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ',
            'σ', 'ς', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        ])
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
        let r = jieba.cut(text, true);
        debug!("jieba cut: {:?}", r);
        for t in r {
            if is_numeric(t) {
                self.push_num_word(t);
            } else if let Some(p) = parse_punctuation(t) {
                self.push_punctuation(p);
            } else if utils::str_is_chinese(t) {
                self.push_zh_word(t);
            } else if t.is_ascii() && !t.trim().is_empty() {
                self.push_en_word(t);
            } else {
                info!("skip word: {:?} in {}", t, text);
            }
        }
    }

    pub fn push_punctuation(&mut self, p: &'static str) {
        match self.sentences.back_mut() {
            Some(Sentence::Zh(zh)) => {
                zh.text.push_str(if p == " " { "," } else { p });
                zh.phones
                    .push(g2pw::G2PWOut::RawChar(p.chars().next().unwrap()));
            }
            Some(Sentence::En(en)) => {
                if p == " "
                    && en
                        .text
                        .last()
                        .map(|w| matches!(w, EnWord::Word(w) if w == "a"))
                        .unwrap_or(false)
                {
                    // Heuristic to handle 'a' as an article followed by a space
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
                log::debug!("skip punctuation: {}", p);
            }
        }
    }

    fn push_en_word(&mut self, word: &str) {
        let word = word.to_ascii_lowercase();
        match self.sentences.back_mut() {
            Some(Sentence::En(en)) => {
                // Try to merge words separated by apostrophes or hyphens
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
                        // If something went wrong, push them back
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
                self.push_en_word(&word)
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
                        zh.phones.push(G2PWOut::Pinyin(p));
                    }
                }
                None => {
                    // Placeholder for words not in the dictionary; G2P will handle them
                    for _ in word.chars() {
                        zh.phones.push(G2PWOut::Pinyin(""));
                    }
                }
            }
        }

        match self.sentences.back_mut() {
            Some(Sentence::Zh(zh)) => {
                h(zh, word);
            }
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
        // Determine language context for number processing
        let lang = match self.sentences.back() {
            Some(Sentence::En(_)) => Lang::En,
            Some(Sentence::Num(n)) => n.lang,
            // Default to Chinese
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
