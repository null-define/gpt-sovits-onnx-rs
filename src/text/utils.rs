use std::collections::HashMap;

use ndarray::{ArrayView, IntoDimension, IxDyn};

use crate::text::g2pw::{MonoChar, PolyChar};
pub static MONO_CHARS_DIST_STR: &str = include_str!("../../resource/g2pw/dict_mono_chars.json");
pub static POLY_CHARS_DIST_STR: &str = include_str!("../../resource/g2pw/dict_poly_chars.json");
pub static LABELS: &str = include_str!("../../resource/g2pw/dict_poly_index_list.json");
pub static DEFAULT_ZH_WORD_DICT: &str = include_str!("../../resource/zh_word_dict.json");



pub fn load_mono_chars() -> HashMap<char, MonoChar> {
    if let Ok(dir) = std::env::var("G2PW_DIST_DIR") {
        let s = std::fs::read_to_string(format!("{}/dict_mono_chars.json", dir))
            .expect("dict_mono_chars.json not found");
        serde_json::from_str(&s).expect("dict_mono_chars.json parse error")
    } else {
        serde_json::from_str(MONO_CHARS_DIST_STR).unwrap()
    }
}

pub fn load_poly_chars() -> HashMap<char, PolyChar> {
    if let Ok(dir) = std::env::var("G2PW_DIST_DIR") {
        let s = std::fs::read_to_string(format!("{}/dict_poly_chars.json", dir))
            .expect("dict_poly_chars.json not found");
        serde_json::from_str(&s).expect("dict_poly_chars.json parse error")
    } else {
        serde_json::from_str(POLY_CHARS_DIST_STR).unwrap()
    }
}

lazy_static::lazy_static! {
    pub static ref DICT_MONO_CHARS: HashMap<char, MonoChar> =load_mono_chars();
    pub static ref DICT_POLY_CHARS: HashMap<char, PolyChar> = load_poly_chars();
    pub static ref POLY_LABLES: Vec<String> = serde_json::from_str(LABELS).unwrap();
}


pub fn str_is_chinese(s: &str) -> bool {
    let mut r = true;
    for c in s.chars() {
        if !DICT_MONO_CHARS.contains_key(&c) && !DICT_POLY_CHARS.contains_key(&c) {
            r &= false;
        }
    }
    r
}

// Finds the index of the maximum value in a 2D tensor
pub fn argmax(tensor: &ArrayView<f32, IxDyn>) -> (usize, usize) {
    let mut max_index = (0, 0);
    let mut max_value = tensor
        .get(IxDyn::zeros(2))
        .copied()
        .unwrap_or(f32::NEG_INFINITY);

    for i in 0..tensor.shape()[0] {
        for j in 0..tensor.shape()[1] {
            if let Some(value) = tensor.get((i, j).into_dimension()) {
                if *value > max_value {
                    max_value = *value;
                    max_index = (i, j);
                }
            }
        }
    }
    max_index
}
