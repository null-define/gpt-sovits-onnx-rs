use log::debug;
use regex::Regex;
use std::collections::HashSet;

use crate::preprocessor::zh::jyutping_list::get_jyutping_list;

// Placeholder for cn2an functionality
fn cn2an_transform(text: &str) -> String {
    // Mock implementation: In real use, integrate a library or custom logic for Arabic numeral to Chinese conversion
    text.to_string()
}

// Mock TextNormalizer struct
struct TextNormalizer;

impl TextNormalizer {
    fn normalize(&self, text: &str) -> Vec<String> {
        // Mock implementation: Split by punctuation and whitespace
        text.split(|c: char| c.is_whitespace() || "，。！？".contains(c))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
}

const INITIALS: &[&str] = &[
    "aa", "aai", "aak", "aap", "aat", "aau", "ai", "au", "ap", "at", "ak", "a", "p", "b", "e",
    "ts", "t", "dz", "d", "kw", "k", "gw", "g", "f", "h", "l", "m", "ng", "n", "s", "y", "w", "c",
    "z", "j", "ong", "on", "ou", "oi", "ok", "o", "uk", "ung", "sp", "spl", "spn", "sil",
];

lazy_static::lazy_static! {
    static ref PUNCTUATION_SET: HashSet<char> = {
        let punctuation = ",.!?;:()[]{}'\"-…";
        punctuation.chars().collect()
    };
}

// Mock ToJyutping functionality
// fn get_jyutping_list(text: &str) -> Vec<(String, String)> {
//     // Mock implementation: In real use, integrate with a Jyutping conversion library
//     // For demo, assume input is "佢個鋤頭太短啦。" and return hardcoded Jyutping
//     vec![
//         ("佢".to_string(), "keoi5".to_string()),
//         ("個".to_string(), "go3".to_string()),
//         ("鋤".to_string(), "co4".to_string()),
//         ("頭".to_string(), "tau4".to_string()),
//         ("太".to_string(), "taai3".to_string()),
//         ("短".to_string(), "dyun2".to_string()),
//         ("啦".to_string(), "laa1".to_string()),
//         ("。".to_string(), ".".to_string()),
//     ]
// }

fn get_jyutping(text: &str) -> Vec<String> {
    let punct_pattern = Regex::new(&format!(
        r"^[{}]+$",
        regex::escape(&PUNCTUATION_SET.iter().collect::<String>())
    ))
    .unwrap();

    let syllables = get_jyutping_list(text);
    debug!("jyutping {:?}", syllables);
    let mut jyutping_array = Vec::new();

    for (word, syllable) in syllables {
        if punct_pattern.is_match(&word) {
            let puncts: Vec<_> = word.chars().map(|c| c.to_string()).collect();
            for punct in puncts {
                if !punct.is_empty() {
                    jyutping_array.push(punct);
                }
            }
        } else {
            // let syllable_pattern = Regex::new(r"^([a-z]+[1-6]+[ ]?)+$").unwrap();
            // if !syllable_pattern.is_match(&syllable) {
            //     panic!("Failed to convert {} to jyutping: {}", word, syllable);
            // }
            jyutping_array.push(syllable);
        }
    }

    jyutping_array
}

fn jyuping_to_initials_finals_tones(jyuping_syllables: Vec<String>) -> (Vec<String>, Vec<i32>) {
    let mut phones = Vec::new();
    let mut word2ph = Vec::new();

    for syllable in jyuping_syllables {
        if PUNCTUATION_SET.contains(&syllable.chars().next().unwrap_or_default()) {
            phones.push(syllable.clone());
            word2ph.push(1);
        } else if syllable == "_" {
            phones.push(syllable.clone());
            word2ph.push(1);
        } else {
            let (tone, syllable_without_tone) =
                if syllable.chars().last().unwrap_or_default().is_digit(10) {
                    let tone = syllable.chars().last().unwrap().to_digit(10).unwrap() as i32;
                    (tone, &syllable[..syllable.len() - 1])
                } else {
                    (0, syllable.as_str())
                };

            let mut found = false;
            for &initial in INITIALS {
                if syllable_without_tone.starts_with(initial) {
                    if syllable_without_tone.starts_with("nga") {
                        let initial_part = &syllable_without_tone[..2];
                        let final_part = if syllable_without_tone[2..].is_empty() {
                            &syllable_without_tone[syllable_without_tone.len() - 1..]
                        } else {
                            &syllable_without_tone[2..]
                        };
                        phones.push(format!("Y{}", initial_part));
                        phones.push(format!("Y{}{}", final_part, tone));
                        word2ph.push(2);
                    } else {
                        let f = if syllable_without_tone[initial.len()..].is_empty() {
                            &initial[initial.len() - 1..]
                        } else {
                            &syllable_without_tone[initial.len()..]
                        };
                        phones.push(format!("Y{}", initial));
                        phones.push(format!("Y{}{}", f, tone));
                        word2ph.push(2);
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                phones.push(format!("Y{}", syllable_without_tone));
                word2ph.push(1);
            }
        }
    }

    (phones, word2ph)
}

pub fn g2p(text: &str) -> (Vec<String>, Vec<i32>) {
    let jyuping = get_jyutping(text);
    debug!("jyuping {:?}", jyuping);
    jyuping_to_initials_finals_tones(jyuping)
}

// fn main() {
//     let text = "佢個鋤頭太短啦。";
//     let normalized_text = text_normalize(text);
//     let (phones, word2ph) = g2p(&normalized_text);
//     println!("Phones: {:?}", phones);
//     println!("Word2Ph: {:?}", word2ph);
// }
