use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::ThreadRng;
use rand::rng;
use std::collections::HashSet;

// Sampling parameters
#[derive(Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,        // Temperature for softmax scaling
    pub top_k: Option<usize>,    // Top-k sampling
    pub top_p: Option<f32>,      // Top-p (nucleus) sampling
    pub repetition_penalty: f32, // Penalty for repeated tokens
}

// Builder for SamplingParams (unchanged)
pub struct SamplingParamsBuilder {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: f32,
}

impl SamplingParamsBuilder {
    pub fn new() -> Self {
        SamplingParamsBuilder {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        }
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        if temperature >= 0.0 {
            self.temperature = temperature;
        } else {
            self.temperature = 1.0;
        }
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        if repetition_penalty > 0.0 {
            self.repetition_penalty = repetition_penalty;
        } else {
            self.repetition_penalty = 1.0;
        }
        self
    }

    pub fn build(self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
        }
    }
}

// Logits sampler struct
pub struct LogitsSampler {
    rng: ThreadRng,
}

impl LogitsSampler {
    pub fn new() -> Self {
        LogitsSampler {
            rng: rng(),
        }
    }

    // Apply repetition penalty to logits based on previous tokens
    fn apply_repetition_penalty(&self, logits: &mut [f32], prev_tokens: &[i64], repetition_penalty: f32) {
        if repetition_penalty != 1.0 {
            let repeated_tokens: HashSet<i64> = prev_tokens.iter().copied().collect();
            for (i, logit) in logits.iter_mut().enumerate() {
                if repeated_tokens.contains(&(i as i64)) {
                    *logit /= repetition_penalty;
                }
            }
        }
    }

    // Apply temperature to logits
    fn apply_temperature(&self, logits: &mut [f32], temperature: f32) {
        if temperature != 1.0 && temperature != 0.0 {
            let inv_temp = 1.0 / temperature;
            for logit in logits.iter_mut() {
                *logit *= inv_temp;
            }
        }
    }

    // Compute softmax probabilities
    fn softmax(&self, logits: &[f32], probs: &mut Vec<f32>) {
        probs.clear();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;
        probs.extend(logits.iter().map(|&x| {
            let exp_val = (x - max_logit).exp();
            sum_exp += exp_val;
            exp_val
        }));
        let inv_sum = 1.0 / sum_exp;
        probs.iter_mut().for_each(|p| *p *= inv_sum);
    }

    // Combined top-k and top-p sampling
    fn filter_indices(&self, probs: &[f32], top_k: Option<usize>, top_p: Option<f32>, avoid_tokens: &[i64]) -> Vec<usize> {
        let avoid_set: HashSet<i64> = avoid_tokens.iter().copied().collect();
        let mut indices: Vec<usize> = (0..probs.len())
            .filter(|&i| !avoid_set.contains(&(i as i64)))
            .collect();

        // Sort indices by probability in descending order
        indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k if specified
        let k = top_k.unwrap_or(indices.len());
        indices.truncate(k);

        // Apply top-p if specified
        if let Some(p) = top_p {
            let mut cumsum = 0.0;
            indices.retain(|&i| {
                cumsum += probs[i];
                cumsum <= p
            });
        }

        indices
    }

    // Argmax sampling with avoidance of specified tokens
    fn argmax(&self, probs: &[f32], avoid_tokens: &[i64]) -> usize {
        let avoid_set: HashSet<i64> = avoid_tokens.iter().copied().collect();
        let mut max_idx = 0;
        let mut max_prob = f32::NEG_INFINITY;
        for (i, &prob) in probs.iter().enumerate() {
            if avoid_set.contains(&(i as i64)) {
                continue;
            }
            if prob > max_prob {
                max_prob = prob;
                max_idx = i;
            }
        }
        max_idx
    }

    // Sample a token with provided parameters
    pub fn sample(&mut self, mut logits: Vec<f32>, prev_tokens: &[i64], params: &SamplingParams, avoid_tokens: &[i64]) -> i64 {
        // Apply repetition penalty
        self.apply_repetition_penalty(&mut logits, prev_tokens, params.repetition_penalty);

        // Apply temperature
        self.apply_temperature(&mut logits, params.temperature);

        // Compute probabilities
        let mut probs = Vec::with_capacity(logits.len());
        self.softmax(&logits, &mut probs);

        // Check if either top_k or top_p is specified
        if params.top_k.is_some() || params.top_p.is_some() {
            let filtered_indices = self.filter_indices(&probs, params.top_k, params.top_p, avoid_tokens);
            if filtered_indices.is_empty() {
                return self.argmax(&probs, avoid_tokens) as i64;
            }
            let filtered_probs: Vec<f32> = filtered_indices.iter().map(|&i| probs[i]).collect();
            let dist = WeightedIndex::new(&filtered_probs).expect("Invalid probability distribution");
            filtered_indices[dist.sample(&mut self.rng)] as i64
        } else {
            self.argmax(&probs, avoid_tokens) as i64
        }
    }
}