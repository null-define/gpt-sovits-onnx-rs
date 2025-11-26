// preprocessor/mod.rs
use anyhow::{Context, Result};
use log::{debug, warn};
use ndarray::{Array2, ArrayBase, Axis, Dim, OwnedRepr};

pub mod bert;
pub mod dict;
pub mod en;
pub mod lang;
pub mod num;
pub mod phone_symbol;
pub mod processor;
pub mod sentence;
pub mod text_normalize;
pub mod text_split;
pub mod utils;
pub mod zh;

pub use lang::{Lang, LangId};
pub use sentence::{Sentence, SentenceProcessor};
pub use text_normalize::text_normalize;
pub use text_split::text_split;

use crate::preprocessor::{
    bert::BertModel,
    en::{EnSentence, g2p_en::G2pEn},
    zh::{
        ZhMode, ZhSentence,
        g2pw::{G2PW, G2PWOut},
    },
};
use jieba_rs::Jieba;
use lang::lang_split;
use processor::{parse_punctuation, push_punctuation};
use std::sync::Arc;

#[derive(Default, Debug)]
struct GroupedSentence {
    text: String,
    word2ph: Vec<i32>,
    phone_ids: Vec<i64>,
    lang: Lang,
}

#[derive(Debug)]
struct SubSentenceData {
    bert_text: String,
    word2ph: Vec<i32>,
    phone_ids: Vec<i64>,
    lang: Lang,
}

pub struct TextProcessor {
    pub jieba: Arc<Jieba>,
    pub g2pw: G2PW,
    pub g2p_en: G2pEn,
    pub bert: BertModel,
}

impl TextProcessor {
    pub fn new(g2pw: G2PW, g2p_en: G2pEn, bert: BertModel) -> Result<Self> {
        Ok(Self {
            jieba: Arc::new(Jieba::new()),
            g2pw,
            g2p_en,
            bert,
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

        let cleaned_text = text_normalize(text);
        debug!("Cleaned text: {}", cleaned_text);
        let chunks = text_split(&cleaned_text);
        let mut result: Vec<(String, Vec<i64>, ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>)> =
            Vec::with_capacity(chunks.len());

        for chunk in chunks.iter() {
            debug!("Processing chunk: {}", chunk);
            let mut sub_sentences = lang_split(chunk, &self.jieba); // Updated call: no sentence_lang param
            if chunk.trim().is_empty() {
                push_punctuation(&mut sub_sentences, ".");
            }
            // --- A. Collect data for all sub-sentences in the chunk ---
            let mut sub_sentences_data: Vec<SubSentenceData> = Vec::new();

            for mut sentence in &mut sub_sentences {
                let g2p_result = match &mut sentence {
                    Sentence::Zh(zh) => {
                        let mode = if matches!(lang_id, LangId::AutoYue) {
                            ZhMode::Cantonese
                        } else {
                            ZhMode::Mandarin
                        };
                        zh.g2p(&mut self.g2pw, mode);
                        zh.build_phone().context("Failed to build Chinese phonemes")
                    }
                    Sentence::En(en) => en
                        .g2p(&mut self.g2p_en)
                        .and_then(|_| en.build_phone().context("Failed to build English phonemes")),
                };

                match g2p_result {
                    Ok(phone_seq) => {
                        if phone_seq.is_empty() {
                            continue; // Skip parts that produce no phonemes
                        }
                        let lang = match &sentence {
                            Sentence::Zh(_) => Lang::Zh,
                            Sentence::En(_) => Lang::En,
                        };
                        sub_sentences_data.push(SubSentenceData {
                            bert_text: sentence.get_text_for_bert(),
                            word2ph: sentence.get_word2ph().to_vec(),
                            phone_ids: sentence.get_phone_ids().to_vec(),
                            lang,
                        });
                    }
                    Err(e) => {
                        warn!("G2P failed for a sentence part in chunk '{}': {}", chunk, e);
                        if cfg!(debug_assertions) {
                            return Err(e.context(format!("G2P failed in chunk: {}", chunk)));
                        }
                        // Continue processing other parts of the chunk
                    }
                }
            }

            // --- B. Group sub-sentences into logically complete sentences, respecting language boundaries ---
            let mut grouped_sentences: Vec<GroupedSentence> = Vec::new();
            let mut current_group: Option<GroupedSentence> = None;

            for data in sub_sentences_data {
                let this_lang = data.lang;
                let ends_sentence = data.bert_text.ends_with('.');

                let mut this_text = data.bert_text.clone();
                let mut this_word2ph = data.word2ph.clone();
                let mut this_phone_ids = data.phone_ids.clone();

                if let Some(mut group) = current_group {
                    if group.lang == this_lang {
                        // Append to current group (same language)
                        // Add space for English if necessary
                        if this_lang == Lang::En
                            && !group.text.ends_with(' ')
                            && !this_text.is_empty()
                        {
                            group.text.push(' ');
                        }
                        group.text.push_str(&this_text);

                        let phone_offset = group.phone_ids.len() as i32;
                        group.phone_ids.extend_from_slice(&this_phone_ids);
                        group
                            .word2ph
                            .extend(this_word2ph.iter().map(|&rel_ph| rel_ph + phone_offset));

                        current_group = Some(group);
                    } else {
                        // Language change: finish current group and start new
                        grouped_sentences.push(group);
                        let new_group = GroupedSentence {
                            text: this_text,
                            word2ph: this_word2ph,
                            phone_ids: this_phone_ids,
                            lang: this_lang,
                        };
                        current_group = Some(new_group);
                    }
                } else {
                    // Start new group
                    let new_group = GroupedSentence {
                        text: this_text,
                        word2ph: this_word2ph,
                        phone_ids: this_phone_ids,
                        lang: this_lang,
                    };
                    current_group = Some(new_group);
                }

                // Check if this completes a sentence
                if ends_sentence {
                    if let Some(group) = current_group.take() {
                        grouped_sentences.push(group);
                    }
                }
            }

            // Add any remaining group
            if let Some(group) = current_group {
                grouped_sentences.push(group);
            }

            // --- C. Process each complete sentence with language-specific BERT ---
            for group in grouped_sentences {
                debug!(
                    "Processing grouped sentence: '{}' (lang: {:?})",
                    group.text, group.lang
                );
                let total_expected_bert_len = group.phone_ids.len();

                match self.bert.get_bert(
                    &group.text,
                    &group.word2ph,
                    total_expected_bert_len,
                    group.lang,
                ) {
                    Ok(bert_features) => {
                        if bert_features.shape()[0] != total_expected_bert_len {
                            let error_msg = format!(
                                "BERT output length mismatch for text '{}': expected {}, got {}",
                                group.text,
                                total_expected_bert_len,
                                bert_features.shape()[0]
                            );
                            warn!("{}", error_msg);
                            if cfg!(debug_assertions) {
                                return Err(anyhow::anyhow!(error_msg));
                            }
                            continue;
                        }
                        if !result.is_empty()
                            && (!result.last().unwrap().0.ends_with('.')
                                || result.last().unwrap().0.trim().len() < 5)
                        {
                            // append to previous
                            let last: &mut (
                                String,
                                Vec<i64>,
                                ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
                            ) = result.last_mut().unwrap();
                            last.0 += &group.text;
                            last.1.extend_from_slice(&group.phone_ids);
                            last.2.append(Axis(0), bert_features.view()).unwrap();
                        } else {
                            result.push((group.text, group.phone_ids, bert_features));
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to get BERT features for text '{}': {}",
                            group.text, e
                        );
                        if cfg!(debug_assertions) {
                            return Err(e.context(format!("BERT failed for text: {}", group.text)));
                        }
                    }
                }
            }
        }

        debug!("RESULT (total sentences: {})", result.len());
        if result.is_empty() {
            return Err(anyhow::anyhow!(
                "No phonemes or BERT features could be generated for the text: {}",
                text
            ));
        }
        Ok(result)
    }
}
