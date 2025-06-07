pub fn is_numeric(text: &str) -> bool {
    text.chars().any(|c| c.is_numeric())
        || text.contains(&['+', '-', '*', '×', '/', '÷', '=', '%'])
        || text.to_lowercase().contains(&[
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ',
            'σ', 'ς', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        ])
}
