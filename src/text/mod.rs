use std::{
    borrow::Cow,
    collections::{HashMap, LinkedList},
    fmt::Debug,
};

use jieba_rs::Jieba;

use log::debug;
use pest::Parser;

pub mod dict;
pub mod g2pw;
pub mod num;
pub mod symbols;

#[inline]
fn get_phone_symbol(symbols: &HashMap<String, i64>, ph: &str) -> i64 {
    // symbols[','] : 3
    symbols.get(ph).map(|id| *id).unwrap_or(3)
}

fn split_zh_ph(ph: &str) -> (&str, &str) {
    match ph {
        "a" => ("AA", "a5"),
        "a1" => ("AA", "a1"),
        "a2" => ("AA", "a2"),
        "a3" => ("AA", "a3"),
        "a4" => ("AA", "a4"),
        "a5" => ("AA", "a5"),

        "ai" => ("AA", "ai5"),
        "ai1" => ("AA", "ai1"),
        "ai2" => ("AA", "ai2"),
        "ai3" => ("AA", "ai3"),
        "ai4" => ("AA", "ai4"),
        "ai5" => ("AA", "ai5"),

        "an" => ("AA", "an5"),
        "an1" => ("AA", "an1"),
        "an2" => ("AA", "an2"),
        "an3" => ("AA", "an3"),
        "an4" => ("AA", "an4"),
        "an5" => ("AA", "an5"),

        "ang" => ("AA", "ang5"),
        "ang1" => ("AA", "ang1"),
        "ang2" => ("AA", "ang2"),
        "ang3" => ("AA", "ang3"),
        "ang4" => ("AA", "ang4"),
        "ang5" => ("AA", "ang5"),

        "ao" => ("AA", "ao5"),
        "ao1" => ("AA", "ao1"),
        "ao2" => ("AA", "ao2"),
        "ao3" => ("AA", "ao3"),
        "ao4" => ("AA", "ao4"),
        "ao5" => ("AA", "ao5"),

        "chi" => ("ch", "ir5"),
        "chi1" => ("ch", "ir1"),
        "chi2" => ("ch", "ir2"),
        "chi3" => ("ch", "ir3"),
        "chi4" => ("ch", "ir4"),
        "chi5" => ("ch", "ir5"),

        "ci" => ("c", "i05"),
        "ci1" => ("c", "i01"),
        "ci2" => ("c", "i02"),
        "ci3" => ("c", "i03"),
        "ci4" => ("c", "i04"),
        "ci5" => ("c", "i05"),

        "e" => ("EE", "e5"),
        "e1" => ("EE", "e1"),
        "e2" => ("EE", "e2"),
        "e3" => ("EE", "e3"),
        "e4" => ("EE", "e4"),
        "e5" => ("EE", "e5"),

        "ei" => ("EE", "ei5"),
        "ei1" => ("EE", "ei1"),
        "ei2" => ("EE", "ei2"),
        "ei3" => ("EE", "ei3"),
        "ei4" => ("EE", "ei4"),
        "ei5" => ("EE", "ei5"),

        "en" => ("EE", "en5"),
        "en1" => ("EE", "en1"),
        "en2" => ("EE", "en2"),
        "en3" => ("EE", "en3"),
        "en4" => ("EE", "en4"),
        "en5" => ("EE", "en5"),

        "eng" => ("EE", "eng5"),
        "eng1" => ("EE", "eng1"),
        "eng2" => ("EE", "eng2"),
        "eng3" => ("EE", "eng3"),
        "eng4" => ("EE", "eng4"),
        "eng5" => ("EE", "eng5"),

        "er" => ("EE", "er5"),
        "er1" => ("EE", "er1"),
        "er2" => ("EE", "er2"),
        "er3" => ("EE", "er3"),
        "er4" => ("EE", "er4"),
        "er5" => ("EE", "er5"),

        "ju" => ("j", "v5"),
        "ju1" => ("j", "v1"),
        "ju2" => ("j", "v2"),
        "ju3" => ("j", "v3"),
        "ju4" => ("j", "v4"),
        "ju5" => ("j", "v5"),

        "juan" => ("j", "van5"),
        "juan1" => ("j", "van1"),
        "juan2" => ("j", "van2"),
        "juan3" => ("j", "van3"),
        "juan4" => ("j", "van4"),
        "juan5" => ("j", "van5"),

        "jue" => ("j", "ve5"),
        "jue1" => ("j", "ve1"),
        "jue2" => ("j", "ve2"),
        "jue3" => ("j", "ve3"),
        "jue4" => ("j", "ve4"),
        "jue5" => ("j", "ve5"),

        "jun" => ("j", "vn5"),
        "jun1" => ("j", "vn1"),
        "jun2" => ("j", "vn2"),
        "jun3" => ("j", "vn3"),
        "jun4" => ("j", "vn4"),
        "jun5" => ("j", "vn5"),

        "o" => ("OO", "o5"),
        "o1" => ("OO", "o1"),
        "o2" => ("OO", "o2"),
        "o3" => ("OO", "o3"),
        "o4" => ("OO", "o4"),
        "o5" => ("OO", "o5"),

        "ou" => ("OO", "ou5"),
        "ou1" => ("OO", "ou1"),
        "ou2" => ("OO", "ou2"),
        "ou3" => ("OO", "ou3"),
        "ou4" => ("OO", "ou4"),
        "ou5" => ("OO", "ou5"),

        "qu" => ("q", "v5"),
        "qu1" => ("q", "v1"),
        "qu2" => ("q", "v2"),
        "qu3" => ("q", "v3"),
        "qu4" => ("q", "v4"),
        "qu5" => ("q", "v5"),

        "quan" => ("q", "van5"),
        "quan1" => ("q", "van1"),
        "quan2" => ("q", "van2"),
        "quan3" => ("q", "van3"),
        "quan4" => ("q", "van4"),
        "quan5" => ("q", "van5"),

        "que" => ("q", "ve5"),
        "que1" => ("q", "ve1"),
        "que2" => ("q", "ve2"),
        "que3" => ("q", "ve3"),
        "que4" => ("q", "ve4"),
        "que5" => ("q", "ve5"),

        "qun" => ("q", "vn5"),
        "qun1" => ("q", "vn1"),
        "qun2" => ("q", "vn2"),
        "qun3" => ("q", "vn3"),
        "qun4" => ("q", "vn4"),
        "qun5" => ("q", "vn5"),

        "ri" => ("r", "ir5"),
        "ri1" => ("r", "ir1"),
        "ri2" => ("r", "ir2"),
        "ri3" => ("r", "ir3"),
        "ri4" => ("r", "ir4"),
        "ri5" => ("r", "ir5"),

        "xu" => ("x", "v5"),
        "xu1" => ("x", "v1"),
        "xu2" => ("x", "v2"),
        "xu3" => ("x", "v3"),
        "xu4" => ("x", "v4"),
        "xu5" => ("x", "v5"),

        "xuan" => ("x", "van5"),
        "xuan1" => ("x", "van1"),
        "xuan2" => ("x", "van2"),
        "xuan3" => ("x", "van3"),
        "xuan4" => ("x", "van4"),
        "xuan5" => ("x", "van5"),

        "xue" => ("x", "ve5"),
        "xue1" => ("x", "ve1"),
        "xue2" => ("x", "ve2"),
        "xue3" => ("x", "ve3"),
        "xue4" => ("x", "ve4"),
        "xue5" => ("x", "ve5"),

        "xun" => ("x", "vn5"),
        "xun1" => ("x", "vn1"),
        "xun2" => ("x", "vn2"),
        "xun3" => ("x", "vn3"),
        "xun4" => ("x", "vn4"),
        "xun5" => ("x", "vn5"),

        "yan" => ("y", "En5"),
        "yan1" => ("y", "En1"),
        "yan2" => ("y", "En2"),
        "yan3" => ("y", "En3"),
        "yan4" => ("y", "En4"),
        "yan5" => ("y", "En5"),

        "ye" => ("y", "E5"),
        "ye1" => ("y", "E1"),
        "ye2" => ("y", "E2"),
        "ye3" => ("y", "E3"),
        "ye4" => ("y", "E4"),
        "ye5" => ("y", "E5"),

        "yu" => ("y", "v5"),
        "yu1" => ("y", "v1"),
        "yu2" => ("y", "v2"),
        "yu3" => ("y", "v3"),
        "yu4" => ("y", "v4"),
        "yu5" => ("y", "v5"),

        "yuan" => ("y", "van5"),
        "yuan1" => ("y", "van1"),
        "yuan2" => ("y", "van2"),
        "yuan3" => ("y", "van3"),
        "yuan4" => ("y", "van4"),
        "yuan5" => ("y", "van5"),

        "yue" => ("y", "ve5"),
        "yue1" => ("y", "ve1"),
        "yue2" => ("y", "ve2"),
        "yue3" => ("y", "ve3"),
        "yue4" => ("y", "ve4"),
        "yue5" => ("y", "ve5"),

        "yun" => ("y", "vn5"),
        "yun1" => ("y", "vn1"),
        "yun2" => ("y", "vn2"),
        "yun3" => ("y", "vn3"),
        "yun4" => ("y", "vn4"),
        "yun5" => ("y", "vn5"),

        "zhi" => ("zh", "ir5"),
        "zhi1" => ("zh", "ir1"),
        "zhi2" => ("zh", "ir2"),
        "zhi3" => ("zh", "ir3"),
        "zhi4" => ("zh", "ir4"),
        "zhi5" => ("zh", "ir5"),

        "zi" => ("z", "i05"),
        "zi1" => ("z", "i01"),
        "zi2" => ("z", "i02"),
        "zi3" => ("z", "i03"),
        "zi4" => ("z", "i04"),
        "zi5" => ("z", "i05"),

        "shi" => ("sh", "ir5"),
        "shi1" => ("sh", "ir1"),
        "shi2" => ("sh", "ir2"),
        "shi3" => ("sh", "ir3"),
        "shi4" => ("sh", "ir4"),
        "shi5" => ("sh", "ir5"),

        "si" => ("s", "i05"),
        "si1" => ("s", "i01"),
        "si2" => ("s", "i02"),
        "si3" => ("s", "i03"),
        "si4" => ("s", "i04"),
        "si5" => ("s", "i05"),

        //['a', 'o', 'e', 'i', 'u', 'ü', 'ai', 'ei', 'ao', 'ou', 'ia', 'ie', 'ua', 'uo', 'üe', 'iao', 'iou', 'uai', 'uei', 'an', 'en', 'ang', 'eng', 'ian', 'in', 'iang', 'ing', 'uan', 'un', 'uang', 'ong', 'üan', 'ün', 'er']
        ph => match split_zh_ph_(ph) {
            (y, "ü") => (y, "v5"),
            (y, "ü1") => (y, "v1"),
            (y, "ü2") => (y, "v2"),
            (y, "ü3") => (y, "v3"),
            (y, "ü4") => (y, "v4"),

            (y, "üe") => (y, "ve5"),
            (y, "üe1") => (y, "ve1"),
            (y, "üe2") => (y, "ve2"),
            (y, "üe3") => (y, "ve3"),
            (y, "üe4") => (y, "ve4"),

            (y, "üan") => (y, "van5"),
            (y, "üan1") => (y, "van1"),
            (y, "üan2") => (y, "van2"),
            (y, "üan3") => (y, "van3"),
            (y, "üan4") => (y, "van4"),

            (y, "ün") => (y, "vn5"),
            (y, "ün1") => (y, "vn1"),
            (y, "ün2") => (y, "vn2"),
            (y, "ün3") => (y, "vn3"),
            (y, "ün4") => (y, "vn4"),

            (y, "a") => (y, "a5"),
            (y, "o") => (y, "o5"),
            (y, "e") => (y, "e5"),
            (y, "i") => (y, "i5"),
            (y, "u") => (y, "u5"),

            (y, "ai") => (y, "ai5"),
            (y, "ei") => (y, "ei5"),
            (y, "ao") => (y, "ao5"),
            (y, "ou") => (y, "ou5"),
            (y, "ia") => (y, "ia5"),
            (y, "ie") => (y, "ie5"),
            (y, "ua") => (y, "ua5"),
            (y, "uo") => (y, "uo5"),
            (y, "iao") => (y, "iao5"),
            (y, "iou") => (y, "iou5"),
            (y, "uai") => (y, "uai5"),
            (y, "uei") => (y, "uei5"),
            (y, "an") => (y, "an5"),
            (y, "en") => (y, "en5"),
            (y, "ang") => (y, "ang5"),
            (y, "eng") => (y, "eng5"),
            (y, "ian") => (y, "ian5"),
            (y, "in") => (y, "in5"),
            (y, "iang") => (y, "iang5"),
            (y, "ing") => (y, "ing5"),
            (y, "uan") => (y, "uan5"),
            (y, "un") => (y, "un5"),
            (y, "uang") => (y, "uang5"),
            (y, "ong") => (y, "ong5"),
            (y, "er") => (y, "er5"),

            (y, s) => (y, s),
        },
    }
}

fn split_zh_ph_(ph: &str) -> (&str, &str) {
    if ph.starts_with("zh") || ph.starts_with("ch") || ph.starts_with("sh") {
        ph.split_at(2)
    } else if ph.starts_with(&[
        'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's',
        'y', 'w',
    ]) {
        // b p m f d t n l g k h j q x r z c s y w

        ph.split_at(1)
    } else {
        (ph, ph)
    }
}

pub struct TextProcessor {
    pub jieba: Jieba,
    pub g2pw: g2pw::G2PWConverter,
    pub symbols: HashMap<String, i64>,
}

impl TextProcessor {
    /// Converts text to phoneme sequences, splitting long sentences to ensure each has >8 words.
    pub fn get_phone(&mut self, text: &str) -> anyhow::Result<Vec<Vec<i64>>> {
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("Input text is empty"));
        }

        // Split text into chunks with >8 words/characters
        let chunks = split_text(text, usize::MAX); // Use large max_chunk_size to rely on word count
        let mut phone_seq = Vec::new();

        for chunk in chunks {
            log::debug!("Processing chunk: {:?}", chunk);

            // Validate chunk size (>8 words for English, >8 characters for Chinese)
            // Build phonemes for the chunk
            let mut phone_builder = PhoneBuilder::new();
            phone_builder.push_text(&self.jieba, &chunk);

            if !chunk.ends_with(['。', '.', '?', '？', '!', '！']) {
                phone_builder.push_punctuation(".");
            }

            for sentence in phone_builder.sentence {
                match sentence {
                    Sentence::Zh(mut zh) => {
                        log::debug!("Processing Zh text: {:?}", zh.zh_text);
                        zh.generate_pinyin(self);
                        match zh.build_phone() {
                            Ok(phones) => phone_seq.push(phones),
                            Err(e) => {
                                log::warn!(
                                    "Failed to build phones for Zh text '{}': {}",
                                    zh.zh_text,
                                    e
                                );
                                if cfg!(debug_assertions) {
                                    return Err(e);
                                }
                            }
                        }
                    }
                    Sentence::En(mut en) => {
                        log::debug!("Processing En text: {:?}", en.en_text);
                        en.generate_phones(self);
                        match en.build_phone() {
                            Ok(phones) => phone_seq.push(phones),
                            Err(e) => {
                                log::warn!(
                                    "Failed to build phones for En text {:?}: {}",
                                    en.en_text,
                                    e
                                );
                                if cfg!(debug_assertions) {
                                    return Err(e);
                                }
                            }
                        }
                    }
                    Sentence::Num(num) => {
                        log::trace!("Processing Num text: {:?}", num.num_text);
                        for s in num.to_phone_sentence()? {
                            match s {
                                Sentence::Zh(mut zh) => {
                                    zh.generate_pinyin(self);
                                    phone_seq.push(zh.build_phone()?);
                                }
                                Sentence::En(mut en) => {
                                    en.generate_phones(self);
                                    phone_seq.push(en.build_phone()?);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        if phone_seq.is_empty() {
            return Err(anyhow::anyhow!("No phonemes generated for text: {}", text));
        }
        Ok(phone_seq)
    }
}

/// Modified split_text to ensure chunks have >8 words/characters
pub fn split_text(text: &str, _max_chunk_size: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    let is_en = text.is_ascii();
    let mut chunks = vec![];
    let mut start_text = text;
    let mut total_count = 0;
    let mut split_index = 0;
    let mut current_chunk = String::new();

    for segment in text.split_inclusive(is_punctuation) {
        let count = if is_en {
            segment.split_whitespace().count()
        } else {
            segment.chars().count()
        };

        log::trace!(
            "segment: {:?}, count: {}, total_count: {}, split_index: {}",
            segment,
            count,
            total_count,
            split_index
        );

        if segment.chars().count() == 1 && is_punctuation(segment.chars().next().unwrap()) {
            current_chunk.push_str(segment);
            split_index += segment.len();
            continue;
        }

        current_chunk.push_str(segment);
        total_count += count;

        if total_count > 8 && (segment.ends_with(['。', '.', '?', '？', '!', '！', '\n'])) {
            let trimmed: &&str = &current_chunk.trim();
            if !trimmed.is_empty() {
                chunks.push(trimmed.to_string());
            }
            start_text = &start_text[split_index..];
            split_index = segment.len();
            total_count = 0;
            current_chunk = String::new();
        } else {
            split_index += segment.len();
        }
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_owned());
    }
    debug!("chunks {:?}", chunks);
    chunks
}

fn is_punctuation(c: char) -> bool {
    ['。', '.', '?', '？', '!', '！', ';', '；', '\n'].contains(&c)
}

#[derive(Debug)]
struct ZhSentence {
    phones_ids: Vec<i64>,
    phones: Vec<g2pw::G2PWOut>,
    word2ph: Vec<i32>,
    zh_text: String,
}

impl ZhSentence {
    fn generate_pinyin(&mut self, processor: &mut TextProcessor) {
        let pinyin = processor
            .g2pw
            .get_pinyin(&self.zh_text)
            .unwrap_or_else(|e| {
                log::warn!("Pinyin generation failed: {}. Using simple plan.", e);
                processor.g2pw.simple_get_pinyin(&self.zh_text)
            });

        if pinyin.len() != self.phones.len() {
            log::warn!(
                "Pinyin length mismatch: {} (pinyin) vs {} (phones)",
                pinyin.len(),
                self.phones.len()
            );
            self.phones = pinyin;
        } else {
            for (i, out) in pinyin.into_iter().enumerate() {
                if matches!(
                    self.phones[i],
                    g2pw::G2PWOut::Pinyin("") | g2pw::G2PWOut::RawChar(_)
                ) {
                    self.phones[i] = out;
                }
            }
        }
        log::debug!("phones: {:?}", self.phones);

        for p in &self.phones {
            match p {
                g2pw::G2PWOut::Pinyin(p) => {
                    let (initial, final_) = split_zh_ph(p);
                    self.phones_ids
                        .push(get_phone_symbol(&processor.symbols, initial));
                    if !final_.is_empty() {
                        self.phones_ids
                            .push(get_phone_symbol(&processor.symbols, final_));
                        self.word2ph.push(2);
                    } else {
                        self.word2ph.push(1);
                    }
                }
                g2pw::G2PWOut::RawChar(c) => {
                    self.phones_ids
                        .push(get_phone_symbol(&processor.symbols, &c.to_string()));
                    self.word2ph.push(1);
                }
            }
        }
    }

    fn build_phone(&self) -> anyhow::Result<Vec<i64>> {
        Ok(self.phones_ids.clone())
    }
}

#[derive(PartialEq, Eq)]
enum EnWord {
    Word(String),
    Punctuation(&'static str),
}

impl Debug for EnWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnWord::Word(w) => write!(f, "\"{}\"", w),
            EnWord::Punctuation(p) => write!(f, "\"{}\"", p),
        }
    }
}

#[derive(Debug)]
struct EnSentence {
    phones_ids: Vec<i64>,
    phones: Vec<Cow<'static, str>>,
    en_text: Vec<EnWord>,
}

impl EnSentence {
    fn generate_phones(&mut self, processor: &TextProcessor) {
        let arpabet = arpabet::load_cmudict();
        for word in &self.en_text {
            match word {
                EnWord::Word(w) => {
                    if let Some(phones) = arpabet.get_polyphone_str(w) {
                        for ph in phones {
                            self.phones.push(Cow::Borrowed(ph));
                            self.phones_ids
                                .push(get_phone_symbol(&processor.symbols, ph));
                        }
                    } else {
                        for c in w.chars() {
                            let c_str = c.to_string();
                            if let Some(phones) = arpabet.get_polyphone_str(&c_str) {
                                for ph in phones {
                                    self.phones.push(Cow::Borrowed(ph));
                                    self.phones_ids
                                        .push(get_phone_symbol(&processor.symbols, ph));
                                }
                            } else {
                                self.phones.push(Cow::Owned(c_str.clone()));
                                self.phones_ids
                                    .push(get_phone_symbol(&processor.symbols, &c_str));
                            }
                        }
                    }
                }
                EnWord::Punctuation(p) => {
                    self.phones.push(Cow::Borrowed(p));
                    self.phones_ids
                        .push(get_phone_symbol(&processor.symbols, p));
                }
            }
        }
        log::debug!("EnSentence phones: {:?}", self.phones);
        log::debug!("EnSentence phones_ids: {:?}", self.phones_ids);
    }

    fn build_phone(&self) -> anyhow::Result<Vec<i64>> {
        Ok(self.phones_ids.clone())
    }
}

#[derive(Debug, Clone, Copy)]
enum Lang {
    Zh,
    En,
}

#[derive(Debug)]
struct NumSentence {
    num_text: String,
    lang: Lang,
}

static NUM_OP: [char; 8] = ['+', '-', '*', '×', '/', '÷', '=', '%'];

impl NumSentence {
    fn need_drop(&self) -> bool {
        let num_text = self.num_text.trim();
        num_text.is_empty() || num_text.chars().all(|c| NUM_OP.contains(&c))
    }

    fn is_link_symbol(&self) -> bool {
        self.num_text == "-"
    }

    fn to_phone_sentence(&self) -> anyhow::Result<LinkedList<Sentence>> {
        // match self.lang {
        //     Lang::Zh => text::num_to_zh_text(symbols, &self.num_text, last_char_is_punctuation),
        //     Lang::En => text::num_to_en_text(symbols, &self.num_text, last_char_is_punctuation),
        // }
        let mut builder = PhoneBuilder::new();
        let pairs = num::ExprParser::parse(num::Rule::all, &self.num_text)?;
        for pair in pairs {
            match self.lang {
                Lang::Zh => num::zh::parse_all(pair, &mut builder)?,
                Lang::En => num::en::parse_all(pair, &mut builder)?,
            }
        }

        Ok(builder.sentence)
    }
}

#[derive(Debug)]
enum Sentence {
    Zh(ZhSentence),
    En(EnSentence),
    Num(NumSentence),
}

#[derive(Debug)]
pub struct PhoneBuilder {
    sentence: LinkedList<Sentence>,
}

fn parse_punctuation(p: &str) -> Option<&'static str> {
    match p {
        "，" | "," => Some(","),
        "。" | "." => Some("."),
        "！" | "!" => Some("!"),
        "？" | "?" => Some("."),
        "；" | ";" => Some("."),
        "：" | ":" => Some(","),
        "‘" | "’" => Some("'"),
        "'" => Some("'"),
        "“" | "”" | "\"" => Some("-"),
        "（" | "(" => Some("-"),
        "）" | ")" => Some("-"),
        "【" | "[" => Some("-"),
        "】" | "]" => Some("-"),
        "《" | "<" => Some("-"),
        "》" | ">" => Some("-"),
        "—" => Some("-"),
        "～" | "~" | "…" | "_" | "..." => Some("…"),
        "·" => Some(","),
        "、" => Some(","),
        "$" => Some("."),
        "/" => Some(","),
        "\n" => Some("."),
        " " => Some(" "),
        // " " => Some("\u{7a7a}"),
        _ => None,
    }
}

fn is_numeric(p: &str) -> bool {
    p.chars().any(|c| c.is_numeric())
        || p.contains(&NUM_OP)
        || p.to_lowercase().contains(&[
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ',
            'σ', 'ς', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        ])
}

impl PhoneBuilder {
    pub fn new() -> Self {
        Self {
            sentence: LinkedList::new(),
        }
    }

    pub fn push_text(&mut self, jieba: &jieba_rs::Jieba, text: &str) {
        let r = jieba.cut(text, true);
        log::info!("jieba cut: {:?}", r);
        for t in r {
            if is_numeric(t) {
                self.push_num_word(t);
            } else if let Some(p) = parse_punctuation(t) {
                self.push_punctuation(p);
            } else if g2pw::str_is_chinese(t) {
                self.push_zh_word(t);
            } else if t.is_ascii() {
                self.push_en_word(t);
            } else {
                log::warn!("skip word: {:?} in {}", t, text);
            }
        }
    }

    pub fn push_punctuation(&mut self, p: &'static str) {
        match self.sentence.back_mut() {
            Some(Sentence::Zh(zh)) => {
                zh.zh_text.push_str(if p == " " { "," } else { p });
                zh.phones
                    .push(g2pw::G2PWOut::RawChar(p.chars().next().unwrap()));
            }
            Some(Sentence::En(en)) => {
                if p == " "
                    && en
                        .en_text
                        .last()
                        .map(|w| match w {
                            EnWord::Word(p) => p == "a",
                            _ => false,
                        })
                        .unwrap_or(false)
                {
                    return;
                }
                en.en_text.push(EnWord::Punctuation(p));
            }
            Some(Sentence::Num(n)) => {
                if n.need_drop() {
                    self.sentence.pop_back();
                }
                self.sentence.push_back(Sentence::En(EnSentence {
                    phones_ids: vec![],
                    phones: vec![],
                    en_text: vec![EnWord::Punctuation(p)],
                }));
            }
            _ => {
                log::debug!("skip punctuation: {}", p);
            }
        }
    }

    pub fn push_en_word(&mut self, word: &str) {
        let word = word.to_ascii_lowercase();
        match self.sentence.back_mut() {
            Some(Sentence::En(en)) => {
                if en
                    .en_text
                    .last()
                    .map(|w| w == &EnWord::Punctuation("'") || w == &EnWord::Punctuation("-"))
                    .unwrap_or(false)
                {
                    let p = en.en_text.pop().unwrap();
                    en.en_text.last_mut().map(|w| {
                        if let EnWord::Word(w) = w {
                            if let EnWord::Punctuation(p) = p {
                                w.push_str(p);
                            }
                            w.push_str(&word);
                        }
                    });
                } else if en
                    .en_text
                    .last()
                    .map(|w| match w {
                        EnWord::Word(w) => w == "a",
                        _ => false,
                    })
                    .unwrap_or(false)
                {
                    en.en_text.last_mut().map(|w| {
                        if let EnWord::Word(w) = w {
                            w.push_str(" ");
                            w.push_str(&word);
                        }
                    });
                } else {
                    en.en_text.push(EnWord::Word(word));
                }
            }
            Some(Sentence::Num(n)) if n.need_drop() => {
                let pop = self.sentence.pop_back().unwrap();
                if let Sentence::Num(n) = pop {
                    if n.is_link_symbol() {
                        self.push_punctuation("-");
                    }
                }
                self.push_en_word(&word)
            }
            _ => {
                let en = EnSentence {
                    phones_ids: vec![],
                    phones: vec![],
                    en_text: vec![EnWord::Word(word)],
                };
                self.sentence.push_back(Sentence::En(en));
            }
        }
    }

    pub fn push_zh_word(&mut self, word: &str) {
        fn h(zh: &mut ZhSentence, word: &str) {
            zh.zh_text.push_str(word);
            match dict::zh_word_dict(word) {
                Some(phones) => {
                    for p in phones {
                        zh.phones.push(g2pw::G2PWOut::Pinyin(p));
                    }
                }
                None => {
                    for _ in word.chars() {
                        zh.phones.push(g2pw::G2PWOut::Pinyin(""));
                    }
                }
            }
        }

        match self.sentence.back_mut() {
            Some(Sentence::Zh(zh)) => {
                h(zh, word);
            }
            Some(Sentence::Num(n)) if n.need_drop() => {
                self.sentence.pop_back();
                self.push_zh_word(word);
            }
            _ => {
                let mut zh = ZhSentence {
                    phones_ids: Vec::new(),
                    phones: Vec::new(),
                    word2ph: Vec::new(),
                    zh_text: String::new(),
                };
                h(&mut zh, word);
                self.sentence.push_back(Sentence::Zh(zh));
            }
        };
    }

    pub fn push_num_word(&mut self, word: &str) {
        match self.sentence.back_mut() {
            Some(Sentence::Zh(_)) => {
                self.sentence.push_back(Sentence::Num(NumSentence {
                    num_text: word.to_string(),
                    lang: Lang::Zh,
                }));
            }
            Some(Sentence::En(_)) => {
                self.sentence.push_back(Sentence::Num(NumSentence {
                    num_text: word.to_string(),
                    lang: Lang::En,
                }));
            }
            Some(Sentence::Num(num)) => {
                num.num_text.push_str(word);
            }
            _ => {
                self.sentence.push_back(Sentence::Num(NumSentence {
                    num_text: word.to_string(),
                    lang: Lang::Zh,
                }));
            }
        }
    }
}
