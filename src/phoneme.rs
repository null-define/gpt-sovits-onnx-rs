pub mod conversion;
pub mod detection;
pub mod symbol;
use conversion::{full_shape_to_half_shape, pinyin_to_phonemes};
use detection::is_numeric;
use symbol::get_phoneme_symbol;
use jieba_rs::Jieba;
use pinyin::ToPinyin;
use std::collections::LinkedList;
use std::fmt::{Debug, Formatter};

pub struct PhonemeExtractor {
    jieba: Jieba,
    sentences: LinkedList<Sentence>,
}

impl Debug for PhonemeExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhonemeExtractor")
            .field("jieba", &"Jieba")
            .field("sentences", &self.sentences)
            .finish()
    }
}

impl Default for PhonemeExtractor {
    fn default() -> Self {
        Self {
            jieba: Default::default(),
            sentences: Default::default(),
        }
    }
}

impl PhonemeExtractor {
    pub fn push_str(&mut self, text: &str) {
        for i in self.jieba.cut(text, true) {
            if is_numeric(i) {
                self.push_num_word(i);
            } else if let Some(i) = full_shape_to_half_shape(i) {
                self.push_punctuation(i);
            } else if i.is_ascii() {
                self.push_en_word(i);
            } else {
                self.push_zh_word(i);
            }
        }
    }

    fn push_en_word(&mut self, word: &str) {
        let arpabet = arpabet::load_cmudict();
        let word = word.to_ascii_lowercase();
        match self.sentences.back_mut() {
            Some(Sentence::En {
                phones,
                phone_ids,
                en_text,
            }) => {
                if let Some(v) = arpabet.get_polyphone_str(&word) {
                    phones.extend_from_slice(&v);
                    for ph in v {
                        phone_ids.push(get_phoneme_symbol(ph));
                    }
                    en_text.push_str(&word);
                } else {
                    for c in word.chars() {
                        let mut b = [0; 4];
                        let c = c.encode_utf8(&mut b);

                        if let Some(v) = arpabet.get_polyphone_str(c) {
                            phones.extend_from_slice(&v);
                            for ph in v {
                                phone_ids.push(get_phoneme_symbol(ph));
                            }
                            en_text.push_str(&c);
                        }
                    }
                }
            }
            _ => {
                if let Some(phones) = arpabet.get_polyphone_str(&word) {
                    let mut phone_ids = vec![];
                    for ph in &phones {
                        phone_ids.push(get_phoneme_symbol(ph));
                    }
                    self.sentences.push_back(Sentence::En {
                        phone_ids,
                        phones,
                        en_text: word.to_string(),
                    });
                } else {
                    let mut phone_ids = vec![];
                    let mut phones = vec![];
                    let mut en_text = String::new();

                    for c in word.chars() {
                        let mut b = [0; 4];
                        let c = c.encode_utf8(&mut b);

                        if let Some(v) = arpabet.get_polyphone_str(c) {
                            phones.extend_from_slice(&v);
                            for ph in v {
                                phone_ids.push(get_phoneme_symbol(ph));
                            }
                            en_text.push_str(&c);
                        }
                    }

                    self.sentences.push_back(Sentence::En {
                        phone_ids,
                        phones,
                        en_text,
                    });
                }
            }
        }
    }

    fn push_zh_word(&mut self, word: &str) {
        match self.sentences.back_mut() {
            Some(Sentence::Zh {
                phone_ids: phones_ids,
                phones,
                word2ph,
                zh_text,
            }) => {
                zh_text.push_str(word);
                for c in word.chars() {
                    if let Some(p) = c.to_pinyin() {
                        let (y, s) = pinyin_to_phonemes(p.with_tone_num_end());
                        phones.push(y);
                        phones_ids.push(get_phoneme_symbol(y));
                        phones.push(s);
                        phones_ids.push(get_phoneme_symbol(s));
                        word2ph.push(2);
                    } else {
                        log::debug!("illegal zh char: {}", c);
                    }
                }
            }
            _ => {
                let mut phones_ids = Vec::new();
                let mut phones = Vec::new();
                let mut word2ph = Vec::new();
                let zh_text = word.to_string();

                for c in word.chars() {
                    if let Some(p) = c.to_pinyin() {
                        let (y, s) = pinyin_to_phonemes(p.with_tone_num_end());
                        phones.push(y);
                        phones_ids.push(get_phoneme_symbol(y));
                        phones.push(s);
                        phones_ids.push(get_phoneme_symbol(s));
                        word2ph.push(2);
                    } else {
                        log::debug!("illegal zh char: {}", c);
                    }
                }

                self.sentences.push_back(Sentence::Zh {
                    phone_ids: phones_ids,
                    phones,
                    word2ph,
                    zh_text,
                });
            }
        }
    }

    fn push_num_word(&mut self, word: &str) {
        match self.sentences.back_mut() {
            Some(Sentence::Zh { .. }) => {
                self.sentences.push_back(Sentence::Num {
                    num_text: word.to_string(),
                    lang: Lang::Zh,
                });
            }
            Some(Sentence::En { .. }) => {
                self.sentences.push_back(Sentence::Num {
                    num_text: word.to_string(),
                    lang: Lang::En,
                });
            }
            Some(Sentence::Num { num_text, .. }) => {
                num_text.push_str(word);
            }
            _ => {
                self.sentences.push_back(Sentence::Num {
                    num_text: word.to_string(),
                    lang: Lang::Zh,
                });
            }
        }
    }

    fn push_punctuation(&mut self, text: &'static str) {
        match self.sentences.back_mut() {
            Some(Sentence::Zh {
                phones,
                phone_ids: phones_ids,
                zh_text,
                word2ph,
            }) => {
                phones.push(text);
                phones_ids.push(get_phoneme_symbol(text));
                zh_text.push_str(text);
                word2ph.push(1);
            }
            Some(Sentence::En {
                phone_ids,
                phones,
                en_text,
            }) => {
                phones.push(text);
                en_text.push_str(text);
                phone_ids.push(get_phoneme_symbol(text));
            }
            Some(Sentence::Num { .. }) => {
                self.sentences.push_back(Sentence::En {
                    phone_ids: vec![get_phoneme_symbol(text)],
                    phones: vec![text],
                    en_text: text.to_string(),
                });
            }
            _ => {
                log::debug!("skip punctuation: {}", text);
            }
        }
    }

    pub fn get_phone_ids(&self) -> Vec<i64> {
        self.sentences
            .iter()
            .flat_map(|i| match i {
                Sentence::Zh { phone_ids, .. } | Sentence::En { phone_ids, .. } => {
                    phone_ids.as_slice()
                }
                Sentence::Num { .. } => Default::default(),
            })
            .map(|i| *i)
            .collect()
    }

    pub fn get_phonemes(&self) -> Vec<&str> {
        self.sentences
            .iter()
            .flat_map(|i| match i {
                Sentence::Zh { phones, .. } | Sentence::En { phones, .. } => phones.as_slice(),
                Sentence::Num { .. } => Default::default(),
            })
            .map(|i| *i)
            .collect()
    }

    pub fn get_word2ph(&self) -> Vec<i32> {
        self.sentences
            .iter()
            .flat_map(|i| match i {
                Sentence::Zh { word2ph, .. } => word2ph.iter().map(|i| *i).collect::<Vec<_>>(),
                Sentence::En { phones, .. } => vec![1; phones.len()],
                Sentence::Num { .. } => vec![1],
            })
            .collect()
    }
}

#[derive(Debug)]
pub enum Sentence {
    Zh {
        phone_ids: Vec<i64>,
        phones: Vec<&'static str>,
        word2ph: Vec<i32>,
        zh_text: String,
    },
    En {
        phone_ids: Vec<i64>,
        phones: Vec<&'static str>,
        en_text: String,
    },
    Num {
        num_text: String,
        lang: Lang,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum Lang {
    Zh,
    En,
}
