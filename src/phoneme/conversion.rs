pub fn full_shape_to_half_shape(text: &str) -> Option<&'static str> {
    match text {
        "，" | "," => Some(","),
        "。" | "." => Some("."),
        "！" | "!" => Some("!"),
        "？" | "?" => Some("?"),
        "；" | ";" => Some(";"),
        "：" | ":" => Some(":"),
        "‘" | "’" | "'" => Some("'"),
        "“" | "”" | "\"" => Some("\""),
        "（" | "(" => Some("("),
        "）" | ")" => Some(")"),
        "【" | "[" => Some("["),
        "】" | "]" => Some("]"),
        "《" | "<" => Some("<"),
        "》" | ">" => Some(">"),
        "—" => Some("-"),
        "～" | "~" | "…" | "_" => Some("…"),
        "·" => Some(","),
        "、" => Some(","),
        "$" => Some("."),
        "/" => Some(","),
        "\n" => Some("."),
        // " " => Some("\u{7a7a}"),
        " " => Some("…"),
        _ => None,
    }
}


pub fn pinyin_to_phonemes(ph: &str) -> (&str, &str) {
    match ph {
        "a" => ("AA", "a"),
        "a1" => ("AA", "a1"),
        "a2" => ("AA", "a2"),
        "a3" => ("AA", "a3"),
        "a4" => ("AA", "a4"),

        "ai" => ("AA", "ai"),
        "ai1" => ("AA", "ai1"),
        "ai2" => ("AA", "ai2"),
        "ai3" => ("AA", "ai3"),
        "ai4" => ("AA", "ai4"),

        "an" => ("AA", "an"),
        "an1" => ("AA", "an1"),
        "an2" => ("AA", "an2"),
        "an3" => ("AA", "an3"),
        "an4" => ("AA", "an4"),

        "ang" => ("AA", "ang"),
        "ang1" => ("AA", "ang1"),
        "ang2" => ("AA", "ang2"),
        "ang3" => ("AA", "ang3"),
        "ang4" => ("AA", "ang4"),

        "ao" => ("AA", "ao"),
        "ao1" => ("AA", "ao1"),
        "ao2" => ("AA", "ao2"),
        "ao3" => ("AA", "ao3"),
        "ao4" => ("AA", "ao4"),

        "chi" => ("ch", "ir"),
        "chi1" => ("ch", "ir1"),
        "chi2" => ("ch", "ir2"),
        "chi3" => ("ch", "ir3"),
        "chi4" => ("ch", "ir4"),

        "ci" => ("c", "i0"),
        "ci1" => ("c", "i01"),
        "ci2" => ("c", "i02"),
        "ci3" => ("c", "i03"),
        "ci4" => ("c", "i04"),

        "e" => ("EE", "e"),
        "e1" => ("EE", "e1"),
        "e2" => ("EE", "e2"),
        "e3" => ("EE", "e3"),
        "e4" => ("EE", "e4"),

        "ei" => ("EE", "ei"),
        "ei1" => ("EE", "ei1"),
        "ei2" => ("EE", "ei2"),
        "ei3" => ("EE", "ei3"),
        "ei4" => ("EE", "ei4"),

        "en" => ("EE", "en"),
        "en1" => ("EE", "en1"),
        "en2" => ("EE", "en2"),
        "en3" => ("EE", "en3"),
        "en4" => ("EE", "en4"),

        "eng" => ("EE", "eng"),
        "eng1" => ("EE", "eng1"),
        "eng2" => ("EE", "eng2"),
        "eng3" => ("EE", "eng3"),
        "eng4" => ("EE", "eng4"),

        "er" => ("EE", "er"),
        "er1" => ("EE", "er1"),
        "er2" => ("EE", "er2"),
        "er3" => ("EE", "er3"),
        "er4" => ("EE", "er4"),

        "ju" => ("j", "v"),
        "ju1" => ("j", "v1"),
        "ju2" => ("j", "v2"),
        "ju3" => ("j", "v3"),
        "ju4" => ("j", "v4"),

        "juan" => ("j", "van"),
        "juan1" => ("j", "van1"),
        "juan2" => ("j", "van2"),
        "juan3" => ("j", "van3"),
        "juan4" => ("j", "van4"),

        "jue" => ("j", "ve"),
        "jue1" => ("j", "ve1"),
        "jue2" => ("j", "ve2"),
        "jue3" => ("j", "ve3"),
        "jue4" => ("j", "ve4"),

        "jun" => ("j", "vn"),
        "jun1" => ("j", "vn1"),
        "jun2" => ("j", "vn2"),
        "jun3" => ("j", "vn3"),
        "jun4" => ("j", "vn4"),

        "o" => ("OO", "o"),
        "o1" => ("OO", "o1"),
        "o2" => ("OO", "o2"),
        "o3" => ("OO", "o3"),
        "o4" => ("OO", "o4"),

        "ou" => ("OO", "ou"),
        "ou1" => ("OO", "ou1"),
        "ou2" => ("OO", "ou2"),
        "ou3" => ("OO", "ou3"),
        "ou4" => ("OO", "ou4"),

        "qu" => ("q", "v"),
        "qu1" => ("q", "v1"),
        "qu2" => ("q", "v2"),
        "qu3" => ("q", "v3"),
        "qu4" => ("q", "v4"),

        "quan" => ("q", "van"),
        "quan1" => ("q", "van1"),
        "quan2" => ("q", "van2"),
        "quan3" => ("q", "van3"),
        "quan4" => ("q", "van4"),

        "que" => ("q", "ve"),
        "que1" => ("q", "ve1"),
        "que2" => ("q", "ve2"),
        "que3" => ("q", "ve3"),
        "que4" => ("q", "ve4"),

        "qun" => ("q", "vn"),
        "qun1" => ("q", "vn1"),
        "qun2" => ("q", "vn2"),
        "qun3" => ("q", "vn3"),
        "qun4" => ("q", "vn4"),

        "ri" => ("r", "ir"),
        "ri1" => ("r", "ir1"),
        "ri2" => ("r", "ir2"),
        "ri3" => ("r", "ir3"),
        "ri4" => ("r", "ir4"),

        "shi" => ("sh", "ir"),
        "shi1" => ("sh", "ir1"),
        "shi2" => ("sh", "ir2"),
        "shi3" => ("sh", "ir3"),
        "shi4" => ("sh", "ir4"),

        "le" => ("l", "E"),
        "le1" => ("l", "E1"),
        "le2" => ("l", "E2"),
        "le3" => ("l", "E3"),
        "le4" => ("l", "E4"),

        "si" => ("s", "i0"),
        "si1" => ("s", "i01"),
        "si2" => ("s", "i02"),
        "si3" => ("s", "i03"),
        "si4" => ("s", "i04"),

        "xu" => ("x", "v"),
        "xu1" => ("x", "v1"),
        "xu2" => ("x", "v2"),
        "xu3" => ("x", "v3"),
        "xu4" => ("x", "v4"),

        "xuan" => ("x", "van"),
        "xuan1" => ("x", "van1"),
        "xuan2" => ("x", "van2"),
        "xuan3" => ("x", "van3"),
        "xuan4" => ("x", "van4"),

        "xue" => ("x", "ve"),
        "xue1" => ("x", "ve1"),
        "xue2" => ("x", "ve2"),
        "xue3" => ("x", "ve3"),
        "xue4" => ("x", "ve4"),

        "xun" => ("x", "vn"),
        "xun1" => ("x", "vn1"),
        "xun2" => ("x", "vn2"),
        "xun3" => ("x", "vn3"),
        "xun4" => ("x", "vn4"),

        "yan" => ("y", "En"),
        "yan1" => ("y", "En1"),
        "yan2" => ("y", "En2"),
        "yan3" => ("y", "En3"),
        "yan4" => ("y", "En4"),

        "ye" => ("y", "E"),
        "ye1" => ("y", "E1"),
        "ye2" => ("y", "E2"),
        "ye3" => ("y", "E3"),
        "ye4" => ("y", "E4"),

        "yu" => ("y", "v"),
        "yu1" => ("y", "v1"),
        "yu2" => ("y", "v2"),
        "yu3" => ("y", "v3"),
        "yu4" => ("y", "v4"),

        "yuan" => ("y", "van"),
        "yuan1" => ("y", "van1"),
        "yuan2" => ("y", "van2"),
        "yuan3" => ("y", "van3"),
        "yuan4" => ("y", "van4"),

        "yue" => ("y", "ve"),
        "yue1" => ("y", "ve1"),
        "yue2" => ("y", "ve2"),
        "yue3" => ("y", "ve3"),
        "yue4" => ("y", "ve4"),

        "yun" => ("y", "vn"),
        "yun1" => ("y", "vn1"),
        "yun2" => ("y", "vn2"),
        "yun3" => ("y", "vn3"),
        "yun4" => ("y", "vn4"),

        "zhi" => ("zh", "ir"),
        "zhi1" => ("zh", "ir1"),
        "zhi2" => ("zh", "ir2"),
        "zhi3" => ("zh", "ir3"),
        "zhi4" => ("zh", "ir4"),

        "zi" => ("z", "i0"),
        "zi1" => ("z", "i01"),
        "zi2" => ("z", "i02"),
        "zi3" => ("z", "i03"),
        "zi4" => ("z", "i04"),
        ph => match split_zh_ph_(ph) {
            (y, "ü1") => (y, "v1"),
            (y, "ü2") => (y, "v2"),
            (y, "ü3") => (y, "v3"),
            (y, "ü4") => (y, "v4"),
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
