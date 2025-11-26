// preprocessor/processor.rs
use crate::preprocessor::{
    dict,
    en::{EnSentence, EnWord},
    num::{NumSentence, is_numeric},
    sentence::Sentence,
    utils::str_is_chinese,
    zh::g2pw::G2PWOut,
};
use log::{debug, warn};

pub fn parse_punctuation(p: &str) -> Option<&'static str> {
    match p {
        "，" | "," => Some(","),
        "。" | "." => Some("."),
        "！" | "!" => Some("!"),
        "？" | "?" => Some("?"),
        "；" | ";" => Some(";"),
        "：" | ":" => Some(":"),
        "‘" | "’" | "'" => Some("'"),
        "＇" => Some("'"),
        "“" | "”" | "\"" => Some("\""),
        "＂" => Some("\""),
        "（" | "(" => Some("("),
        "）" | ")" => Some(")"),
        "【" | "[" => Some("["),
        "】" | "]" => Some("]"),
        "《" | "<" => Some("<"),
        "》" | ">" => Some(">"),
        "—" | "–" => Some("-"),
        "～" | "~" => Some("~"),
        "…" | "..." => Some("..."),
        "·" => Some("·"),
        "、" => Some("、"),
        "$" => Some("$"),
        "/" => Some("/"),
        "\n" => Some("\n"),
        " " => Some(" "),
        _ => None,
    }
}

pub fn push_punctuation(sentences: &mut Vec<Sentence>, p: &'static str) {
    match sentences.last_mut() {
        Some(Sentence::Zh(zh)) => {
            zh.text.push_str(p);
            if let Some(c) = p.chars().next() {
                zh.phones.push(G2PWOut::RawChar(c));
            }
        }
        Some(Sentence::En(en)) => {
            if p == " " && matches!(en.text.last(), Some(EnWord::Word(w)) if w == "a") {
                return;
            }
            en.text.push(EnWord::Punctuation(p));
        }
        None => {
            let en = EnSentence {
                phone_ids: vec![],
                phones: vec![],
                text: vec![EnWord::Punctuation(p)],
                word2ph: vec![],
            };
            sentences.push(Sentence::En(en));
        }
    }
}

pub fn push_en_word(sentences: &mut Vec<Sentence>, word: &str) {
    if word.ends_with('.') {
        let en = EnSentence {
            phone_ids: vec![],
            phones: vec![],
            text: vec![EnWord::Word(word.to_string())],
            word2ph: vec![],
        };
        sentences.push(Sentence::En(en));
        return;
    }
    match sentences.last_mut() {
        Some(Sentence::En(en)) => {
            if matches!(en.text.last(), Some(EnWord::Punctuation(p)) if *p == "'" || *p == "-") {
                let p = en.text.pop().unwrap();
                if let Some(EnWord::Word(last_word)) = en.text.last_mut() {
                    if let EnWord::Punctuation(p_str) = p {
                        last_word.push_str(&p_str);
                        last_word.push_str(word);
                        return;
                    }
                }
                en.text.push(p);
            }
            en.text.push(EnWord::Word(word.to_string()));
        }
        _ => {
            let en = EnSentence {
                phone_ids: vec![],
                phones: vec![],
                text: vec![EnWord::Word(word.to_string())],
                word2ph: vec![],
            };
            sentences.push(Sentence::En(en));
        }
    }
}

pub fn push_zh_word(sentences: &mut Vec<Sentence>, word: &str) {
    fn add_zh_word(zh: &mut crate::preprocessor::zh::ZhSentence, word: &str) {
        zh.text.push_str(word);
        match dict::zh_word_dict(word) {
            Some(phones) => {
                zh.phones.extend(
                    phones
                        .into_iter()
                        .map(|p: &String| G2PWOut::Pinyin(p.clone())),
                );
            }
            None => {
                zh.phones
                    .extend(word.chars().map(|_| G2PWOut::Pinyin(String::new())));
            }
        }
    }

    if word.ends_with('.') {
        let mut zh = crate::preprocessor::zh::ZhSentence {
            phone_ids: vec![],
            phones: vec![],
            word2ph: vec![],
            text: String::new(),
        };
        add_zh_word(&mut zh, word);
        sentences.push(Sentence::Zh(zh));
        return;
    }

    match sentences.last_mut() {
        Some(Sentence::Zh(zh)) => add_zh_word(zh, word),
        _ => {
            let mut zh = crate::preprocessor::zh::ZhSentence {
                phone_ids: vec![],
                phones: vec![],
                word2ph: vec![],
                text: String::new(),
            };
            add_zh_word(&mut zh, word);
            sentences.push(Sentence::Zh(zh));
        }
    }
}

/// Processes a single token based on language, handling numbers, Chinese, English, or mixed.
pub fn lang_process_token(
    sentences: &mut Vec<Sentence>,
    t: &str,
    sentence_lang: crate::preprocessor::Lang,
) {
    if t.trim().is_empty() {
        return;
    }
    if let Some(p) = parse_punctuation(t) {
        push_punctuation(sentences, p);
        return;
    }

    if is_numeric(t) {
        let ns = NumSentence {
            text: t.to_owned(),
            lang: sentence_lang,
        };
        let txt = match ns.to_lang_text() {
            Ok(txt) => txt,
            Err(e) => {
                warn!("Failed to process numeric token '{}': {}", t, e);
                t.to_string()
            }
        };
        match sentence_lang {
            crate::preprocessor::Lang::Zh => push_zh_word(sentences, &txt),
            crate::preprocessor::Lang::En => push_en_word(sentences, &txt),
        }
        return;
    } else if str_is_chinese(t) {
        push_zh_word(sentences, t);
        return;
    } else if t
        .chars()
        .all(|c| c.is_ascii_alphabetic() || c == '\'' || c == '-')
    {
        push_en_word(sentences, t);
        return;
    }

    // Handle mixed-language tokens by re-tokenizing the mixed token iteratively
    lang_mixup(sentences, t, sentence_lang);
}

/// Handles mixed-language tokens by re-tokenizing and processing sub-tokens iteratively.
pub fn lang_mixup(
    sentences: &mut Vec<Sentence>,
    mixed_token: &str,
    sentence_lang: crate::preprocessor::Lang,
) {
    // Replace recursion with iteration to prevent stack overflow
    let mut tokens_to_process = vec![mixed_token];

    while let Some(sub_token) = tokens_to_process.pop() {
        for sub_token_match in crate::preprocessor::lang::TOKEN_REGEX.find_iter(sub_token) {
            let sub_token_str = sub_token_match.as_str();
            lang_process_token(sentences, sub_token_str, sentence_lang);
        }
    }
}
