use fancy_regex::{Captures, Regex};
use log::{debug, error, info};

pub struct TextPreprocessor {
    pub min_word_count: usize,
}

impl TextPreprocessor {
    pub(crate) fn init() -> Self {
        TextPreprocessor { min_word_count: 16 }
    }

    pub fn process(&self, sentence: String) -> Vec<String> {
        let mut sentences: Vec<String> = vec![];
        let mut text = sentence;

        // Check if text is primarily Chinese (more than 50% Chinese characters)
        let total_chars = text.chars().count();
        let chinese_count = text
            .chars()
            .filter(|c| {
                ('\u{4E00}'..='\u{9FFF}').contains(c) // Basic Chinese character range
            })
            .count();
        let is_mostly_chinese =
            total_chars > 0 && (chinese_count as f32 / total_chars as f32) > 0.5;

        // Count non-whitespace characters for word count
        let word_count: usize = text.chars().filter(|c| !c.is_whitespace()).count();

        // If below min_word_count, return the trimmed text as a single sentence
        if word_count < self.min_word_count {
            if !text.trim().is_empty() {
                sentences.push(text.trim().to_string());
            }
            return sentences;
        }

        // Replace special characters with commas
        text = Regex::new(r"[——《》【】<=>{}()（）#&@“”^_|…\\]")
            .unwrap()
            .replace_all(&text, ",")
            .to_string();
        while text.contains(",,") {
            text = text.replace(",,", ",");
        }

        // If mostly Chinese, split by character; otherwise, split by sentence-ending punctuation
        if is_mostly_chinese {
            // For Chinese text, split by character if needed
            let chars: Vec<char> = text.chars().collect();
            let mut current_sentence = String::new();
            let mut current_count = 0;

            for c in chars {
                current_sentence.push(c);
                if !c.is_whitespace() {
                    current_count += 1;
                }
                // Split when reaching min_word_count or sentence-ending punctuation
                if (current_count >= self.min_word_count
                    && [',', '。', '；', '！', '？'].contains(&c))
                    || current_count >= self.min_word_count * 2
                // Fallback for long sequences
                {
                    if !current_sentence.trim().is_empty() {
                        debug!("current sentence {}", current_sentence);
                        sentences.push(current_sentence.trim().to_string());
                    }
                    current_sentence = String::new();
                    current_count = 0;
                }
            }
            // Add remaining text if not empty
            if !current_sentence.trim().is_empty() {
                debug!("current sentence {}", current_sentence);
                current_sentence = self.normalize_sentence(&current_sentence);
                sentences.push(current_sentence.trim().to_string());
            }
        } else {
            // For non-Chinese text, split by sentence-ending punctuation
            let re = Regex::new(r"[;.?!]").unwrap();
            let mut last_pos = 0;
            let mut current_count = 0;
            let mut current_sentence = String::new();

            for mat in re.find_iter(&text) {
                let end = mat.unwrap().end();
                let segment = &text[last_pos..end];
                let segment_count = segment.chars().filter(|c| !c.is_whitespace()).count();

                current_sentence.push_str(segment);
                current_count += segment_count;

                // Only split if accumulated text meets min_word_count
                if current_count >= self.min_word_count {
                    debug!("current sentence {}", current_sentence);
                    if !current_sentence.trim().is_empty() {
                        sentences.push(current_sentence.trim().to_string());
                    }
                    current_sentence = String::new();
                    current_count = 0;
                }
                last_pos = end;
            }

            // Add remaining text if not empty
            let remaining = &text[last_pos..];
            if !remaining.trim().is_empty() {
                current_sentence.push_str(remaining);
                if !current_sentence.trim().is_empty() {
                    sentences.push(current_sentence.trim().to_string());
                }
            }
        }
        sentences
    }

    fn _post_replace(&self, sentence: String) -> String {
        let mut sentence = sentence;
        sentence = sentence.replace("/", "每");
        sentence = sentence.replace("~", "至");
        sentence = sentence.replace("～", "至");
        sentence = sentence.replace("①", "一");
        sentence = sentence.replace("②", "二");
        sentence = sentence.replace("③", "三");
        sentence = sentence.replace("④", "四");
        sentence = sentence.replace("⑤", "五");
        sentence = sentence.replace("⑥", "六");
        sentence = sentence.replace("⑦", "七");
        sentence = sentence.replace("⑧", "八");
        sentence = sentence.replace("⑨", "九");
        sentence = sentence.replace("⑩", "十");
        sentence = sentence.replace("α", "阿尔法");
        sentence = sentence.replace("β", "贝塔");
        sentence = sentence.replace("γ", "伽玛").replace("Γ", "伽玛");
        sentence = sentence.replace("δ", "德尔塔").replace("Δ", "德尔塔");
        sentence = sentence.replace("ε", "艾普西龙");
        sentence = sentence.replace("ζ", "捷塔");
        sentence = sentence.replace("η", "依塔");
        sentence = sentence.replace("θ", "西塔").replace("Θ", "西塔");
        sentence = sentence.replace("ι", "艾欧塔");
        sentence = sentence.replace("κ", "喀帕");
        sentence = sentence.replace("λ", "拉姆达").replace("Λ", "拉姆达");
        sentence = sentence.replace("μ", "缪");
        sentence = sentence.replace("ν", "拗");
        sentence = sentence.replace("ξ", "克西").replace("Ξ", "克西");
        sentence = sentence.replace("ο", "欧米克伦");
        sentence = sentence.replace("π", "派").replace("Π", "派");
        sentence = sentence.replace("ρ", "肉");
        sentence = sentence
            .replace("ς", "西格玛")
            .replace("Σ", "西格玛")
            .replace("σ", "西格玛");
        sentence = sentence.replace("τ", "套");
        sentence = sentence.replace("υ", "宇普西龙");
        sentence = sentence.replace("φ", "服艾").replace("Φ", "服艾");
        sentence = sentence.replace("χ", "器");
        sentence = sentence.replace("ψ", "普赛").replace("Ψ", "普赛");
        sentence = sentence.replace("ω", "欧米伽").replace("Ω", "欧米伽");

        sentence = Regex::new(r"[-——《》【】<=>{}()（）#&@“”^_|…\\]")
            .unwrap()
            .replace_all(&sentence, "")
            .to_string();

        sentence
    }

    fn normalize_sentence(&self, sentence: &str) -> String {
        if sentence.trim().is_empty() {
            return String::new();
        }
        let mut sentence = sentence.to_string();
        // check for input is english or chinese, if chinese, do this.
        sentence = self._post_replace(sentence);

        sentence
    }
}
