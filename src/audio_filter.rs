use hound::{SampleFormat, WavSpec, WavWriter};
use log::{debug};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::Arc;
use std::time::Instant;
use std::{env, fs::File, io::BufWriter, path::Path};

// Trait for filter passes
pub trait FilterPass: Send + Sync {
    fn process(&mut self, buffer: &mut [f32]);
    fn name(&self) -> &str;
}

// Noise reduction filter using spectral subtraction with rustfft
pub struct NoiseReductionPass {
    name: String,
    noise_profile: Vec<f32>, // Magnitude spectrum of noise
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>, // Hann window
    fft_buffer: Vec<Complex<f32>>,
    ifft_buffer: Vec<Complex<f32>>,
    // Re-usable buffer to avoid allocations in the process loop
    frame_input: Vec<f32>,
    output_buffer: Vec<f32>,
    output: Vec<f32>,
}

impl NoiseReductionPass {
    pub fn new(sample_rate: u32, fft_size: usize) -> Self {
        let hop_size = fft_size / 4; // 75% overlap for smooth reconstruction
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // Create Hann window
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos()))
            .collect();

        // Initialize noise profile (will be updated during processing)
        let noise_profile = vec![0.0; fft_size / 2 + 1];

        NoiseReductionPass {
            name: "NoiseReduction".to_string(),
            noise_profile,
            fft_size,
            hop_size,
            sample_rate,
            fft,
            ifft,
            window,
            fft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            ifft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            frame_input: vec![0.0; fft_size],
            output_buffer: vec![0.0; fft_size],
            output: Vec::new(),
        }
    }

    // Process a single frame
    fn process_frame(&mut self) {
        // Apply window and copy to FFT buffer
        for (i, (sample, window)) in self.frame_input.iter().zip(self.window.iter()).enumerate() {
            self.fft_buffer[i] = Complex::new(sample * window, 0.0);
        }

        // Forward FFT
        self.fft.process(&mut self.fft_buffer);

        // Update noise profile (assume first few frames are noise-only for simplicity)
        for (i, bin) in self
            .fft_buffer
            .iter()
            .enumerate()
            .take(self.noise_profile.len())
        {
            self.noise_profile[i] = 0.9 * self.noise_profile[i] + 0.1 * bin.norm();
        }

        // Apply spectral subtraction
        for (i, bin) in self
            .fft_buffer
            .iter_mut()
            .enumerate()
            .take(self.noise_profile.len())
        {
            let magnitude = bin.norm();
            let new_magnitude = (magnitude - self.noise_profile[i]).max(0.0);
            let scale = if magnitude > 0.0 {
                new_magnitude / magnitude
            } else {
                0.0
            };
            *bin = bin.scale(scale);
        }

        // Inverse FFT
        self.ifft_buffer.copy_from_slice(&self.fft_buffer);
        self.ifft.process(&mut self.ifft_buffer);

        // Copy to output buffer with windowing
        for (i, (bin, window)) in self.ifft_buffer.iter().zip(self.window.iter()).enumerate() {
            self.output_buffer[i] = bin.re * window / self.fft_size as f32;
        }
    }
}

impl FilterPass for NoiseReductionPass {
    fn process(&mut self, buffer: &mut [f32]) {
        // Ensure the output buffer is large enough
        if self.output.len() < buffer.len() {
            self.output.resize(buffer.len(), 0.0);
        }
        // Clear the output buffer for the new run
        self.output.iter_mut().for_each(|s| *s = 0.0);

        // Process frames with overlap-add
        for frame_start in (0..buffer.len()).step_by(self.hop_size) {
            let frame_end = (frame_start + self.fft_size).min(buffer.len());
            let frame = &buffer[frame_start..frame_end];

            // Clear the frame_input buffer
            self.frame_input.iter_mut().for_each(|s| *s = 0.0);
            self.frame_input[..frame.len()].copy_from_slice(frame);

            self.process_frame();

            // Overlap-add
            for (i, &sample) in self.output_buffer.iter().enumerate() {
                if frame_start + i < self.output.len() {
                    self.output[frame_start + i] += sample;
                }
            }
            if frame_end == buffer.len() {
                break;
            }
        }

        // Normalize output (account for windowing gain)
        let window_gain = self.window.iter().map(|&w| w * w).sum::<f32>() / self.hop_size as f32;
        for sample in self.output.iter_mut() {
            *sample /= window_gain;
        }

        buffer.copy_from_slice(&self.output[..buffer.len()]);
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// High-pass filter pass (first-order)
pub struct HighPassPass {
    name: String,
    alpha: f32, // Smoothing factor based on cutoff frequency
    prev_input: f32,
    prev_output: f32,
}

impl HighPassPass {
    pub fn new(sample_rate: u32, cutoff_freq: f32) -> Self {
        // Calculate alpha for first-order high-pass filter
        // RC = 1 / (2 * pi * cutoff_freq), alpha = RC / (RC + dt)
        let rc = 1.0 / (2.0 * PI * cutoff_freq);
        let dt = 1.0 / sample_rate as f32;
        let alpha = rc / (rc + dt);

        HighPassPass {
            name: "HighPass".to_string(),
            alpha,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }
}

impl FilterPass for HighPassPass {
    fn process(&mut self, buffer: &mut [f32]) {
        // Apply first-order high-pass filter: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        for sample in buffer.iter_mut() {
            let output = self.alpha * (self.prev_output + *sample - self.prev_input);
            self.prev_input = *sample;
            self.prev_output = output;
            *sample = output.clamp(-1.0, 1.0); // Prevent clipping
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Equalizer pass to enhance human speech (FFT-based)
pub struct EqualizerPass {
    name: String,
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>, // Hann window
    fft_buffer: Vec<Complex<f32>>,
    ifft_buffer: Vec<Complex<f32>>,
    // Re-usable buffer to avoid allocations in the process loop
    frame_input: Vec<f32>,
    output_buffer: Vec<f32>,
    output: Vec<f32>,
}

impl EqualizerPass {
    pub fn new(sample_rate: u32, fft_size: usize) -> Self {
        let hop_size = fft_size / 4; // 75% overlap
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // Create Hann window
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos()))
            .collect();

        EqualizerPass {
            name: "Equalizer".to_string(),
            fft_size,
            hop_size,
            sample_rate,
            fft,
            ifft,
            window,
            fft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            ifft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            frame_input: vec![0.0; fft_size],
            output_buffer: vec![0.0; fft_size],
            output: Vec::new(),
        }
    }

    // Process a single frame
    fn process_frame(&mut self) {
        // Apply window and copy to FFT buffer
        for (i, (sample, window)) in self.frame_input.iter().zip(self.window.iter()).enumerate() {
            self.fft_buffer[i] = Complex::new(sample * window, 0.0);
        }

        // Forward FFT
        self.fft.process(&mut self.fft_buffer);

        // Apply equalization
        let freq_per_bin = self.sample_rate as f32 / self.fft_size as f32;
        for (i, bin) in self
            .fft_buffer
            .iter_mut()
            .enumerate()
            .take(self.fft_size / 2 + 1)
        {
            let freq = i as f32 * freq_per_bin;
            let gain = if freq < 100.0 {
                0.5 // -6 dB for low frequencies
            } else if (100.0..=2000.0).contains(&freq) {
                2.0 // +6 dB for speech range
            } else if freq > 8000.0 {
                0.5 // -6 dB for high frequencies
            } else {
                1.0 // No change outside target ranges
            };
            *bin = bin.scale(gain);
        }
        // Inverse FFT
        self.ifft_buffer.copy_from_slice(&self.fft_buffer);
        self.ifft.process(&mut self.ifft_buffer);

        // Copy to output buffer with windowing
        for (i, (bin, window)) in self.ifft_buffer.iter().zip(self.window.iter()).enumerate() {
            self.output_buffer[i] = bin.re * window / self.fft_size as f32;
        }
    }
}

impl FilterPass for EqualizerPass {
    fn process(&mut self, buffer: &mut [f32]) {
        // Ensure the output buffer is large enough
        if self.output.len() < buffer.len() {
            self.output.resize(buffer.len(), 0.0);
        }
        self.output.iter_mut().for_each(|s| *s = 0.0);

        // Process frames with overlap-add
        for frame_start in (0..buffer.len()).step_by(self.hop_size) {
            let frame_end = (frame_start + self.fft_size).min(buffer.len());
            let frame = &buffer[frame_start..frame_end];

            self.frame_input.iter_mut().for_each(|s| *s = 0.0);
            self.frame_input[..frame.len()].copy_from_slice(frame);

            self.process_frame();

            // Overlap-add
            for (i, &sample) in self.output_buffer.iter().enumerate() {
                if frame_start + i < self.output.len() {
                    self.output[frame_start + i] += sample;
                }
            }
            if frame_end == buffer.len() {
                break;
            }
        }

        // Normalize output (account for windowing gain)
        let window_gain = self.window.iter().map(|&w| w * w).sum::<f32>() / self.hop_size as f32;
        for sample in self.output.iter_mut() {
            *sample /= window_gain;
            *sample = sample.clamp(-1.0, 1.0); // Prevent clipping
        }

        buffer.copy_from_slice(&self.output[..buffer.len()]);
    }

    fn name(&self) -> &str {
        &self.name
    }
}
// Reverb pass to add natural ambiance
pub struct ReverbPass {
    name: String,
    impulse_response: Vec<f32>,
    history: Vec<f32>,
    history_index: usize,
}

impl ReverbPass {
    pub fn new(sample_rate: u32) -> Self {
        // Generate a synthetic impulse response for a small room
        let ir_length = (sample_rate as f32 * 0.3) as usize; // 300 ms reverb tail
        let mut impulse_response = vec![0.0; ir_length];

        // Early reflections (simplified: 3 discrete echoes)
        impulse_response[0] = 1.0; // Direct sound
        impulse_response[(sample_rate as f32 * 0.01) as usize] = 0.6; // 10 ms reflection
        impulse_response[(sample_rate as f32 * 0.02) as usize] = 0.4; // 20 ms reflection
        impulse_response[(sample_rate as f32 * 0.03) as usize] = 0.2; // 30 ms reflection

        // Exponential decay tail
        let decay_time = 0.5; // RT60 decay time in seconds
        let decay_factor = (-3.0 / (decay_time * sample_rate as f32)).exp();
        for i in 0..ir_length {
            let t = i as f32 / sample_rate as f32;
            impulse_response[i] *= decay_factor.powf(t * sample_rate as f32);
            // Add slight random modulation for naturalness
            impulse_response[i] += (rand::random::<f32>() - 0.5) * 0.01 * impulse_response[i].abs();
        }

        // Normalize IR to prevent gain increase
        let ir_sum: f32 = impulse_response.iter().map(|x| x.abs()).sum();
        if ir_sum > 1.0 {
            impulse_response.iter_mut().for_each(|x| *x /= ir_sum);
        }

        ReverbPass {
            name: "Reverb".to_string(),
            impulse_response,
            history: vec![0.0; ir_length],
            history_index: 0,
        }
    }
}

impl FilterPass for ReverbPass {
    fn process(&mut self, buffer: &mut [f32]) {
        let mut output = vec![0.0; buffer.len()];
        let ir = &self.impulse_response;
        let ir_len = ir.len();

        // Perform time-domain convolution with wet/dry mix
        for (i, &sample) in buffer.iter().enumerate() {
            // Update history buffer
            self.history[self.history_index] = sample;

            // Convolve with IR
            let mut sum = 0.0;
            for j in 0..ir_len {
                let hist_idx = (self.history_index + ir_len - j) % ir_len;
                sum += self.history[hist_idx] * ir[j];
            }

            // Wet/dry mix: 20% reverb, 80% dry signal
            output[i] = sample * 0.8 + sum * 0.2;

            // Update history index
            self.history_index = (self.history_index + 1) % ir_len;
        }

        // Copy output to buffer with clipping prevention
        for (out, &sample) in buffer.iter_mut().zip(output.iter()) {
            *out = sample.clamp(-1.0, 1.0);
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Plugin system to manage filter passes
pub struct AudioFilterPluginSystem {
    spec: WavSpec,
    passes: Vec<Box<dyn FilterPass>>,
}

impl AudioFilterPluginSystem {
    pub fn new(spec: WavSpec) -> Self {
        AudioFilterPluginSystem {
            spec,
            passes: Vec::new(),
        }
    }

    // Add a filter pass dynamically
    pub fn add_pass(&mut self, pass: Box<dyn FilterPass>) {
        debug!("Adding pass: {}", pass.name());
        self.passes.push(pass);
    }

    // Process the audio buffer through all passes
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        let total_start_time = Instant::now();
        for pass in self.passes.iter_mut() {
            let pass_start_time = Instant::now();
            pass.process(buffer);
            let pass_duration = pass_start_time.elapsed();
            debug!("Pass '{}' took: {:?}", pass.name(), pass_duration);
        }
        let total_duration = total_start_time.elapsed();
        debug!("------------------------------------");
        debug!("Total processing time: {:?}", total_duration);
    }
}
