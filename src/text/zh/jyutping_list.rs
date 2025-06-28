use regex::Regex;
use serde_json;
use std::collections::HashMap;

lazy_static::lazy_static! {
    static ref TONED: HashMap<u32, String> = load_data();
    static ref TONELESS: HashMap<u32, String> = load_data();
}

fn load_data() -> HashMap<u32, String> {
    let mut toned = HashMap::new();
    let mut toneless = HashMap::new();
    let contents = include_str!("../../../resource/jyutping_dictionary.json");

    let table: HashMap<String, String> =
        serde_json::from_str(&contents).expect("Failed to parse JSON");

    for (code, jyutping) in table {
        let jyutping = jyutping.split_whitespace().next().unwrap_or("").to_string();
        let code_int = u32::from_str_radix(&code, 16).expect("Failed to parse code as u32");
        toned.insert(code_int, format!(" {} ", jyutping));
        toneless.insert(
            code_int,
            format!(" {} ", jyutping[..jyutping.len() - 1].to_string()),
        );
    }

    // Return toned for TONED and toneless for TONELESS
    if std::env::var("TONELESS").is_ok() {
        toneless
    } else {
        toned
    }
}

pub fn get_jyutping(text: &str, tone: bool) -> String {
    if text.is_empty() {
        return (String::new());
    }

    let re = Regex::new(r"[\u{4e00}-\u{9fff}]+")
        .map_err(|e| format!("Regex error: {}", e))
        .unwrap();
    if !re.is_match(text) {
        return (text.to_string());
    }

    let dict = if tone { &*TONED } else { &*TONELESS };
    let mut converted = String::new();

    for c in text.chars() {
        let code = c as u32;
        if let Some(jyutping) = dict.get(&code) {
            converted.push_str(jyutping);
        } else {
            converted.push(c);
        }
    }

    // Join and trim multiple spaces
    let result: String = converted
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");
    (result)
}

pub fn get_jyutping_list(text: &str) -> Vec<(String, String)> {
    let mut res = Vec::new();
    for ch in text.chars() {
        res.push((ch.to_string(), get_jyutping(&ch.to_string(), false)));
    }
    res
}
