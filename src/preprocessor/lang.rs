// preprocessor/lang.rs
use crate::preprocessor::processor;
use crate::preprocessor::sentence::Sentence;
use crate::preprocessor::utils::{is_numeric_or_punctuation, str_is_chinese, str_is_numeric};
use jieba_rs::Jieba;
use log::debug;
use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lang {
    Zh,
    En,
}

impl Default for Lang {
    fn default() -> Self {
        Lang::Zh
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LangId {
    Auto,    // Mandarin
    AutoYue, // Cantonese
}

/// Updates the function to support Chinese numerals and mixed Chinese/number segments
fn split_into_lang_segments(text: &str) -> Vec<(Lang, &str)> {
    let mut segments = Vec::new();
    let mut byte_offset = 0usize;
    let bytes = text.as_bytes();
    let mut prev_lang = Lang::Zh; // Default to Chinese

    while byte_offset < bytes.len() {
        // Find start of the next segment
        let char_start = text[byte_offset..].chars().next().unwrap_or('\0');
        let is_zh_start = str_is_chinese(&char_start.to_string());

        // If the segment is a Chinese numeral and previous segment was Chinese, treat it as part of Chinese
        let mut byte_end = byte_offset;
        for ch in text[byte_offset..].chars() {
            // Check if it's a Chinese numeral or Chinese character
            if (str_is_chinese(&ch.to_string())) != is_zh_start {
                break;
            }
            let ch_bytes = ch.len_utf8();
            byte_end += ch_bytes;
        }

        let segment = &text[byte_offset..byte_end];
        let lang = if is_zh_start { Lang::Zh } else { Lang::En };

        // If the previous segment was Chinese and this is a number, we should treat it as Chinese as well
        if prev_lang == Lang::Zh && is_numeric_or_punctuation(segment) {
            prev_lang = Lang::Zh;
        } else {
            prev_lang = lang;
        }

        if !segment.trim().is_empty() {
            segments.push((prev_lang, segment));
        }

        byte_offset = byte_end;
    }

    segments
}

/// Splits text into language-specific sentences using language segments.
/// Tokenizes each segment according to its language (Jieba for Chinese, regex for English),
/// then processes tokens into Sentences. This allows splitting by language while composing
/// back within the same sentence (segments are processed into Sentences, then grouped later).
pub fn lang_split(text: &str, jieba: &Jieba) -> Vec<Sentence> {
    let segments = split_into_lang_segments(text);
    let mut sentences = Vec::with_capacity(16);
    debug!("Lang segments: {:?}", segments);

    for (seg_lang, seg_text) in segments {
        if seg_text.trim().is_empty() {
            continue;
        }

        let words: Vec<&str> = match seg_lang {
            Lang::Zh => jieba.cut(seg_text, true).into_iter().collect(),
            Lang::En => vec![seg_text],
        };

        if words.is_empty() {
            continue;
        }

        // Process words for this segment into one Sentence (composed back)
        for word in words {
            processor::lang_process_token(&mut sentences, word, seg_lang);
        }
    }

    sentences
}

// Simplified regex for tokenization
pub(crate) static TOKEN_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?x)
        [\p{Han}]+ |              # Chinese characters
        [a-zA-Z]+(?:['-][a-zA-Z]+)* | # English words with optional apostrophes/hyphens
        \d+(?:\.\d+)? |          # Numbers (including decimals)
        [.,!?;:()\[\]<>\-\"$/\u{3001}\u{3002}\u{FF01}\u{FF1F}\u{FF1B}\u{FF1A}\u{FF0C}\u{2018}\u{2019}\u{201C}\u{201D}] | # Punctuation
        \s+                      # Whitespace
        "#,
    )
    .unwrap()
});
