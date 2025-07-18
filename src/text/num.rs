use std::collections::LinkedList;

use anyhow::Result;
use pest::Parser;

use crate::text::Lang;

#[derive(pest_derive::Parser)]
#[grammar = "resource/rule.pest"]
pub struct ExprParser;

pub mod zh {

    use super::*;

    use pest::iterators::Pair;

    fn parse_pn(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::pn);
        match pair.as_str() {
            "+" => dst_string.push_str("加"),
            "-" => dst_string.push_str("减"),
            "*" | "×" => dst_string.push_str("乘"),
            "/" | "÷" => dst_string.push_str("除以"),
            "=" => dst_string.push_str("等于"),
            _ => {
                #[cfg(debug_assertions)]
                unreachable!("unknown: {:?} in pn", pair);
            }
        }
        Ok(())
    }

    fn parse_flag(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::flag);
        match pair.as_str() {
            "+" => dst_string.push_str("正"),
            "-" => dst_string.push_str("负"),
            _ => {
                #[cfg(debug_assertions)]
                unreachable!("unknown: {:?} in flag", pair);
            }
        }
        Ok(())
    }

    fn parse_percent(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::percent);
        // percent = { (decimals|integer)~"%" }

        dst_string.push_str("百分之");
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::decimals => parse_decimals(pair, dst_string)?,
                Rule::integer => {
                    parse_integer(pair, dst_string, true)?;
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in expr", pair.as_str());
                }
            }
        }
        Ok(())
    }

    static UNITS: [&str; 4] = ["", "十", "百", "千"];
    static BASE_UNITS: [&str; 4] = ["", "万", "亿", "万"];

    fn parse_integer(
        pair: Pair<Rule>,
        dst_string: &mut String,
        unit: bool,
    ) -> anyhow::Result<LinkedList<(&'static str, &'static str)>> {
        assert_eq!(pair.as_rule(), Rule::integer);

        let mut r: LinkedList<(&str, &str)> = LinkedList::new();

        let inner = pair.into_inner().rev();
        let mut n = 0;

        for pair in inner {
            let txt = match pair.as_str() {
                "0" => "零",
                "1" => "一",
                "2" => "二",
                "3" => "三",
                "4" => "四",
                "5" => "五",
                "6" => "六",
                "7" => "七",
                "8" => "八",
                "9" => "九",
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in integer", n);
                    #[cfg(not(debug_assertions))]
                    ""
                }
            };
            let u = if n % 4 != 0 {
                UNITS[n % 4]
            } else {
                BASE_UNITS[(n / 4) % 4]
            };

            r.push_front((txt, u));
            n += 1;
        }

        if r.iter().all(|(s, _)| s == &"零") {
            dst_string.push_str("零");
            return Ok(r);
        }

        if unit {
            let mut last_is_zero = true;
            for (s, u) in &r {
                if last_is_zero && s == &"零" {
                    continue;
                }
                if s == &"零" {
                    dst_string.push_str(s);
                    last_is_zero = true;
                } else {
                    dst_string.push_str(s);
                    dst_string.push_str(u);
                    last_is_zero = false;
                }
            }
        } else {
            for (s, _) in &r {
                dst_string.push_str(s);
            }
        }

        Ok(r)
    }

    fn parse_decimals(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::decimals);

        let mut inner = pair.into_inner().rev();
        let f_part = inner.next().unwrap();
        if let Some(i_part) = inner.next() {
            parse_integer(i_part, dst_string, true)?;
        } else {
            dst_string.push_str("零");
        }
        dst_string.push_str("点");
        parse_integer(f_part, dst_string, false)?;

        Ok(())
    }

    fn parse_fractional(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::fractional);

        let mut inner = pair.into_inner();
        let numerator = inner.next().unwrap();
        let denominator = inner.next().unwrap();
        parse_integer(denominator, dst_string, true)?;
        dst_string.push_str("分之");
        parse_integer(numerator, dst_string, true)?;
        Ok(())
    }

    fn parse_num(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::num);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::flag => parse_flag(pair, dst_string)?,
                Rule::percent => parse_percent(pair, dst_string)?,
                Rule::decimals => {
                    parse_decimals(pair, dst_string)?;
                }
                Rule::fractional => {
                    parse_fractional(pair, dst_string)?;
                }
                Rule::integer => {
                    parse_integer(pair, dst_string, true)?;
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in num", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_signs(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::signs);

        let inner = pair.into_inner();
        for pair in inner {
            log::debug!("{:?}", pair);
            match pair.as_rule() {
                Rule::num => parse_num(pair, dst_string)?,
                Rule::pn => parse_pn(pair, dst_string)?,
                Rule::word => {
                    log::warn!("word: {:?}", pair.as_str());
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in expr", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_link(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::link);

        match pair.as_str() {
            "-" => dst_string.push_str("杠"),
            _ => dst_string.push_str(""),
        }

        Ok(())
    }

    fn parse_word(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::word);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::digit => {
                    let txt = match pair.as_str() {
                        "0" => "零",
                        "1" => "一",
                        "2" => "二",
                        "3" => "三",
                        "4" => "四",
                        "5" => "五",
                        "6" => "六",
                        "7" => "七",
                        "8" => "八",
                        "9" => "九",
                        n => {
                            #[cfg(debug_assertions)]
                            unreachable!("unknown: {:?} in integer", n);

                            #[cfg(not(debug_assertions))]
                            n
                        }
                    };
                    dst_string.push_str(txt);
                }
                Rule::alpha => {
                    let txt = pair.as_str();
                    dst_string.push_str(txt);
                }
                Rule::greek => {
                    let txt = match pair.as_str() {
                        "α" | "Α" => "阿尔法",
                        "β" | "Β" => "贝塔",
                        "γ" | "Γ" => "伽马",
                        "δ" | "Δ" => "德尔塔",
                        "ε" | "Ε" => "艾普西龙",
                        "ζ" | "Ζ" => "泽塔",
                        "η" | "Η" => "艾塔",
                        "θ" | "Θ" => "西塔",
                        "ι" | "Ι" => "约塔",
                        "κ" | "Κ" => "卡帕",
                        "λ" | "Λ" => "兰姆达",
                        "μ" | "Μ" => "缪",
                        "ν" | "Ν" => "纽",
                        "ξ" | "Ξ" => "克西",
                        "ο" | "Ο" => "欧米克戈",
                        "π" | "Π" => "派",
                        "ρ" | "Ρ" => "罗",
                        "σ" | "Σ" => "西格玛",
                        "τ" | "Τ" => "套",
                        "υ" | "Υ" => "宇普西龙",
                        "φ" | "Φ" => "斐",
                        "χ" | "Χ" => "希",
                        "ψ" | "Ψ" => "普西",
                        "ω" | "Ω" => "欧米伽",
                        _ => {
                            #[cfg(debug_assertions)]
                            unreachable!("unknown: {:?} in greek", pair.as_str());
                            #[cfg(not(debug_assertions))]
                            ""
                        }
                    };
                    dst_string.push_str(txt);
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in word", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_ident(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::ident);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::word => parse_word(pair, dst_string)?,
                Rule::link => parse_link(pair, dst_string)?,
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in ident", pair.as_str());
                }
            }
        }
        Ok(())
    }

    pub fn parse_all(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::all);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::signs => parse_signs(pair, dst_string)?,
                Rule::ident => parse_ident(pair, dst_string)?,
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in all", pair.as_str());
                }
            }
        }
        Ok(())
    }
}

pub mod en {
    use super::*;

    use pest::iterators::Pair;

    const SEPARATOR: &'static str = " ";

    fn parse_pn(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::pn);
        match pair.as_str() {
            "+" => dst_string.push_str("plus"),
            "-" => dst_string.push_str("minus"),
            "*" | "×" => dst_string.push_str("times"),
            "/" | "÷" => {
                dst_string.push_str("divided");
                dst_string.push_str("by");
            }
            "=" => dst_string.push_str("is"),
            _ => {
                #[cfg(debug_assertions)]
                unreachable!("unknown: {:?} in pn", pair);
            }
        }

        Ok(())
    }

    fn parse_flag(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::flag);
        match pair.as_str() {
            "-" => dst_string.push_str("negative"),
            _ => {
                #[cfg(debug_assertions)]
                unreachable!("unknown: {:?} in flag", pair);
            }
        }
        Ok(())
    }

    fn parse_percent(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::percent);
        // percent = { (decimals|integer)~"%" }

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::decimals => parse_decimals(pair, dst_string)?,
                Rule::integer => {
                    parse_integer(pair, dst_string, true)?;
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in expr", pair.as_str());
                }
            }
        }
        dst_string.push_str("percent");

        Ok(())
    }

    fn parse_integer(pair: Pair<Rule>, dst_string: &mut String, unit: bool) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::integer);
        if unit {
            if let Ok(r) = num2en::str_to_words(pair.as_str()) {
                r.split(&[' ', '-']).for_each(|s| {
                    dst_string.push_str(s);
                });
                return Ok(());
            }
        }

        let inner = pair.into_inner();
        for pair in inner {
            let txt = match pair.as_str() {
                "0" => "zero",
                "1" => "one",
                "2" => "two",
                "3" => "three",
                "4" => "four",
                "5" => "five",
                "6" => "six",
                "7" => "seven",
                "8" => "eight",
                "9" => "nine",
                n => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in integer", n);
                    #[cfg(not(debug_assertions))]
                    n
                }
            };
            dst_string.push_str(txt);
        }

        Ok(())
    }

    fn parse_decimals(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::decimals);
        if let Ok(r) = num2en::str_to_words(pair.as_str()) {
            r.split(&[' ', '-']).for_each(|s| {
                dst_string.push_str(s);
            });
            return Ok(());
        }

        let mut inner = pair.into_inner().rev();
        let f_part = inner.next().unwrap();
        if let Some(i_part) = inner.next() {
            parse_integer(i_part, dst_string, true)?;
        } else {
            dst_string.push_str("zero");
        }
        dst_string.push_str("point");

        parse_integer(f_part, dst_string, false)?;

        Ok(())
    }

    fn parse_fractional(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::fractional);

        let mut inner = pair.into_inner();
        let numerator = inner.next().unwrap();
        let denominator = inner.next().unwrap();
        parse_integer(numerator, dst_string, true)?;
        dst_string.push_str("over");
        parse_integer(denominator, dst_string, true)?;

        Ok(())
    }

    fn parse_num(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::num);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::flag => parse_flag(pair, dst_string)?,
                Rule::percent => parse_percent(pair, dst_string)?,
                Rule::decimals => {
                    parse_decimals(pair, dst_string)?;
                }
                Rule::fractional => {
                    parse_fractional(pair, dst_string)?;
                }
                Rule::integer => {
                    parse_integer(pair, dst_string, true)?;
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in num", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_signs(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::signs);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::num => parse_num(pair, dst_string)?,
                Rule::pn => parse_pn(pair, dst_string)?,
                Rule::word => {
                    println!("word: {:?}", pair.as_str());
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in expr", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_link(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::link);

        Ok(())
    }

    fn parse_word(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::word);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::digit => {
                    let txt = match pair.as_str() {
                        "0" => "zero",
                        "1" => "one",
                        "2" => "two",
                        "3" => "three",
                        "4" => "four",
                        "5" => "five",
                        "6" => "six",
                        "7" => "seven",
                        "8" => "eight",
                        "9" => "nine",
                        n => {
                            #[cfg(debug_assertions)]
                            unreachable!("unknown: {:?} in integer", n);
                            #[cfg(not(debug_assertions))]
                            n
                        }
                    };
                    dst_string.push_str(txt);
                }
                Rule::alpha => {
                    let txt = pair.as_str();
                    dst_string.push_str(txt);
                }
                Rule::greek => {
                    let txt = match pair.as_str() {
                        "α" | "Α" => "alpha",
                        "β" | "Β" => "beta",
                        "γ" | "Γ" => "gamma",
                        "δ" | "Δ" => "delta",
                        "ε" | "Ε" => "epsilon",
                        "ζ" | "Ζ" => "zeta",
                        "η" | "Η" => "eta",
                        "θ" | "Θ" => "theta",
                        "ι" | "Ι" => "iota",
                        "κ" | "Κ" => "kappa",
                        "λ" | "Λ" => "lambda",
                        "μ" | "Μ" => "mu",
                        "ν" | "Ν" => "nu",
                        "ξ" | "Ξ" => "xi",
                        "ο" | "Ο" => "omicron",
                        "π" | "Π" => "pi",
                        "ρ" | "Ρ" => "rho",
                        "σ" | "Σ" => "sigma",
                        "τ" | "Τ" => "tau",
                        "υ" | "Υ" => "upsilon",
                        "φ" | "Φ" => "phi",
                        "χ" | "Χ" => "chi",
                        "ψ" | "Ψ" => "psi",
                        "ω" | "Ω" => "omega",

                        _ => {
                            #[cfg(debug_assertions)]
                            unreachable!("unknown: {:?} in greek", pair.as_str());
                            #[cfg(not(debug_assertions))]
                            ""
                        }
                    };
                    dst_string.push_str(txt);
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in word", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_ident(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::ident);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::word => parse_word(pair, dst_string)?,
                Rule::link => parse_link(pair, dst_string)?,
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in ident", pair.as_str());
                }
            }
        }
        Ok(())
    }

    pub fn parse_all(pair: Pair<Rule>, dst_string: &mut String) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::all);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::signs => parse_signs(pair, dst_string)?,
                Rule::ident => parse_ident(pair, dst_string)?,
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in all", pair.as_str());
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct NumSentence {
    pub text: String,
    pub lang: Lang,
}

static NUM_OP: [char; 8] = ['+', '-', '*', '×', '/', '÷', '=', '%'];

impl NumSentence {
    pub fn need_drop(&self) -> bool {
        let num_text = self.text.trim();
        num_text.is_empty() || num_text.chars().all(|c| NUM_OP.contains(&c))
    }

    pub fn is_link_symbol(&self) -> bool {
        self.text == "-"
    }

    pub fn to_lang_text(&self) -> Result<String> {
        let mut dst_string = String::new();
        let pairs = ExprParser::parse(Rule::all, &self.text)?;
        for pair in pairs {
            match self.lang {
                Lang::Zh => zh::parse_all(pair, &mut dst_string)?,
                Lang::En => en::parse_all(pair, &mut dst_string)?,
            }
        }
        Ok(dst_string)
    }
}

pub fn is_numeric(p: &str) -> bool {
    p.chars().any(|c| c.is_numeric())
        || p.contains(&NUM_OP)
        || p.to_lowercase().contains(&[
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ',
            'σ', 'ς', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        ])
}
