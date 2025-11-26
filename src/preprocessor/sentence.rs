// preprocessor/sentence.rs
use crate::{
    preprocessor::bert::BertModel, preprocessor::en::EnSentence, preprocessor::zh::ZhSentence,
};
use log::debug;

#[derive(Debug)]
pub enum Sentence {
    Zh(ZhSentence),
    En(EnSentence),
}

pub trait SentenceProcessor {
    fn get_text_for_bert(&self) -> String;
    fn get_word2ph(&self) -> &[i32];
    fn get_phone_ids(&self) -> &[i64];
}

impl SentenceProcessor for EnSentence {
    fn get_text_for_bert(&self) -> String {
        let mut result = String::with_capacity(self.text.len() * 10);
        for word in &self.text {
            match word {
                crate::preprocessor::en::EnWord::Word(w) => {
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                    result.push_str(w);
                }
                crate::preprocessor::en::EnWord::Punctuation(p) => {
                    result.push_str(p);
                }
            }
        }
        debug!("English BERT text: {}", result);
        result
    }

    fn get_word2ph(&self) -> &[i32] {
        &self.word2ph
    }

    fn get_phone_ids(&self) -> &[i64] {
        &self.phone_ids
    }
}

impl SentenceProcessor for ZhSentence {
    fn get_text_for_bert(&self) -> String {
        debug!("Chinese BERT text: {}", self.text);
        self.text.clone()
    }

    fn get_word2ph(&self) -> &[i32] {
        &self.word2ph
    }

    fn get_phone_ids(&self) -> &[i64] {
        &self.phone_ids
    }
}

impl SentenceProcessor for Sentence {
    fn get_text_for_bert(&self) -> String {
        match self {
            Sentence::Zh(zh) => zh.get_text_for_bert(),
            Sentence::En(en) => en.get_text_for_bert(),
        }
    }

    fn get_word2ph(&self) -> &[i32] {
        match self {
            Sentence::Zh(zh) => zh.get_word2ph(),
            Sentence::En(en) => en.get_word2ph(),
        }
    }

    fn get_phone_ids(&self) -> &[i64] {
        match self {
            Sentence::Zh(s) => s.get_phone_ids(),
            Sentence::En(s) => s.get_phone_ids(),
        }
    }
}
