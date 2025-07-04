// Static list of consonants for split_zh_ph_
static CONSONANTS: &[char] = &[
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y',
    'w',
];

// Optimized phoneme splitting function for Chinese
// NOTE: This large match statement could be further optimized using a HashMap or a phf::Map
// if performance profiling indicates it as a bottleneck.
pub fn split_zh_ph(ph: &str) -> (&str, &str) {
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

// Helper function to split phonemes based on consonants
#[inline]
pub fn split_zh_ph_(ph: &str) -> (&str, &str) {
    if ph.starts_with("zh") || ph.starts_with("ch") || ph.starts_with("sh") {
        ph.split_at(2)
    } else if ph.chars().next().map_or(false, |c| CONSONANTS.contains(&c)) {
        ph.split_at(1)
    } else {
        (ph, ph)
    }
}
