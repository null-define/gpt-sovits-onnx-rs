// preprocessor/text_normalize.rs
use once_cell::sync::Lazy;
use regex::Regex;

/// Filters out emojis and other non-essential symbols from the text.
pub fn text_normalize(text: &str) -> String {
    let temp = CLEANUP_REGEX.replace_all(text, " ").into_owned();
    let temp = PUNCTUATION_COMMAS_REGEX.replace_all(&temp, ",");
    // Then replace all other punctuation with a period
    PUNCTUATION_PERIODS_REGEX
        .replace_all(&temp, ".")
        .into_owned()
}

// Regex to handle emojis and symbols
static CLEANUP_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{2600}-\u{27BF}\u{2000}-\u{206F}\u{2300}-\u{23FF}]+",
    )
    .unwrap()
});

// Expanded regex to match punctuation marks that should be replaced with a period
static PUNCTUATION_PERIODS_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"[\u{2026}\u{003F}\u{0021}\u{002E}\u{FF01}\u{FF1F}\u{3002}\u{FF0E}]+", // Ellipsis, ?, !, ., Chinese ?, Chinese full stop
    )
    .unwrap()
});

// Regex to match other punctuation marks (including #) that should be replaced with a comma
static PUNCTUATION_COMMAS_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"[\u{002C}\u{2018}\u{2019}\u{201C}\u{201D}\u{2022}\u{FF0C}\u{FF1A}\u{FF1B}\u{FF0B}\u{FF1D}\u{FF5E}\u{2014}\u{2013}\u{FF3B}\u{FF3D}\u{FF08}\u{FF09}\u{3001}\u{FF5F}\u{FF1C}\u{FF1E}\u{300A}\u{300B}\u{300C}\u{300D}\u{FF1F}\u{FF3F}\u{002A}\u{003D}\u{00A9}\u{2212}\u{2021}\u{203B}\u{2047}\u{3008}\u{3009}\u{300E}\u{300F}\u{FF0F}\u{0023}]+", // Added # (hash) here
    )
    .unwrap()
});

// Regex to collapse multiple commas or periods to a single one
static COLLAPSE_COMMAS_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"([,])\1+").unwrap() // Match multiple consecutive commas (`,`)
});

static COLLAPSE_PERIODS_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"([.])\1+").unwrap() // Match multiple consecutive periods (`.`)
});
