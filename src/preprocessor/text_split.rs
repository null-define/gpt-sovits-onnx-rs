// preprocessor/text_split.rs
use crate::preprocessor::{Lang, utils::str_is_chinese};

/// Splits the input text into sentences based on end punctuation, handling newlines and abbreviations.
pub fn text_split(text: &str) -> Vec<String> {
    let mut sentences = Vec::with_capacity(text.len() / 20);
    let mut current = String::with_capacity(64);
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        // Handle newlines separately - don't add them to current sentence
        if c == '\n' || c == '\r' {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
            continue;
        }

        current.push(c);

        // Check if current character is end punctuation
        let is_end_punctuation = matches!(c, '.' | '\n');

        if is_end_punctuation {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }

    // Handle any remaining text
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}
