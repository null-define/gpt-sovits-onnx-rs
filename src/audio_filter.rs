use hound::{SampleFormat, WavSpec};
use log::debug;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use smallvec::SmallVec;
use std::f32::consts::PI;
use std::sync::Arc;
use std::time::Instant;

// Trait for filter passes
pub trait FilterPass: Send + Sync {
    fn process(&mut self, buffer: &mut [f32]);
    fn name(&self) -> &str;
}

// High-pass filter pass
pub struct HighPassPass {
    name: String,
    alpha: f32,
    prev_input: f32,
    prev_output: f32,
}

impl HighPassPass {
    pub fn new(sample_rate: u32, cutoff_freq: f32) -> Self {
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
        for sample in buffer.iter_mut() {
            let output = self.alpha * (self.prev_output + *sample - self.prev_input);
            self.prev_input = *sample;
            self.prev_output = output;
            *sample = output.clamp(-1.0, 1.0);
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Reverb pass with FFT-based convolution
pub struct ReverbPass {
    name: String,
    impulse_response: Vec<Complex<f32>>,
    fft_size: usize,
    hop_size: usize,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    window_gain: f32,
    fft_buffer: Vec<Complex<f32>>,
    frame_input: SmallVec<[f32; 1024]>,
    output_buffer: SmallVec<[f32; 1024]>,
}

impl ReverbPass {
    pub fn new(sample_rate: u32) -> Self {
        let fft_size = (sample_rate as f32 * 0.3) as usize;
        let hop_size = fft_size / 4;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        let mut impulse_response = vec![0.0; fft_size];
        impulse_response[0] = 1.0;
        impulse_response[(sample_rate as f32 * 0.01) as usize] = 0.6;
        impulse_response[(sample_rate as f32 * 0.02) as usize] = 0.4;
        impulse_response[(sample_rate as f32 * 0.03) as usize] = 0.2;

        let decay_time = 0.5;
        let decay_factor = (-3.0 / (decay_time * sample_rate as f32)).exp();
        for i in 0..fft_size {
            let t = i as f32 / sample_rate as f32;
            impulse_response[i] *= decay_factor.powf(t * sample_rate as f32);
            impulse_response[i] += (rand::random::<f32>() - 0.5) * 0.01 * impulse_response[i].abs();
        }
        let ir_sum: f32 = impulse_response.iter().map(|x| x.abs()).sum();
        if ir_sum > 1.0 {
            impulse_response.iter_mut().for_each(|x| *x /= ir_sum);
        }

        let mut ir_fft = impulse_response
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect::<Vec<_>>();
        fft.process(&mut ir_fft);

        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos()))
            .collect();
        let window_gain = window.iter().map(|&w| w * w).sum::<f32>() / hop_size as f32;

        ReverbPass {
            name: "Reverb".to_string(),
            impulse_response: ir_fft,
            fft_size,
            hop_size,
            fft,
            ifft,
            window,
            window_gain,
            fft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            frame_input: SmallVec::from_vec(vec![0.0; fft_size]),
            output_buffer: SmallVec::from_vec(vec![0.0; fft_size]),
        }
    }

    fn process_frame(&mut self) {
        self.fft_buffer
            .iter_mut()
            .zip(self.frame_input.iter().zip(self.window.iter()))
            .for_each(|(b, (&s, &w))| {
                *b = Complex::new(s * w, 0.0);
            });
        self.fft.process(&mut self.fft_buffer);
        self.fft_buffer
            .iter_mut()
            .zip(self.impulse_response.iter())
            .for_each(|(b, &ir)| {
                *b *= ir;
            });
        self.ifft.process(&mut self.fft_buffer);
        self.fft_buffer
            .iter()
            .zip(self.window.iter())
            .enumerate()
            .for_each(|(i, (bin, &w))| {
                self.output_buffer[i] = bin.re * w / self.fft_size as f32;
            });
    }
}

impl FilterPass for ReverbPass {
    fn process(&mut self, buffer: &mut [f32]) {
        let mut output = vec![0.0; buffer.len()];
        for frame_start in (0..buffer.len()).step_by(self.hop_size) {
            let frame_end = (frame_start + self.fft_size).min(buffer.len());
            let frame = &buffer[frame_start..frame_end];
            self.frame_input.iter_mut().for_each(|s| *s = 0.0);
            self.frame_input[..frame.len()].copy_from_slice(frame);
            self.process_frame();
            for (i, &sample) in self.output_buffer.iter().enumerate() {
                if frame_start + i < output.len() {
                    output[frame_start + i] += sample;
                }
            }
            if frame_end == buffer.len() {
                break;
            }
        }

        for (out, &dry) in output.iter_mut().zip(buffer.iter()) {
            *out = (*out / self.window_gain) * 0.2 + dry * 0.8;
            *out = out.clamp(-1.0, 1.0);
        }
        buffer.copy_from_slice(&output[..buffer.len()]);
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Plugin system to manage filter passes
pub struct AudioFilterPluginSystem {
    spec: WavSpec,
    passes: Vec<Box<dyn FilterPass>>,
    shared_buffer: Vec<f32>,
}

impl AudioFilterPluginSystem {
    pub fn new(spec: WavSpec) -> Self {
        AudioFilterPluginSystem {
            spec,
            passes: Vec::new(),
            shared_buffer: Vec::new(),
        }
    }

    pub fn add_pass(&mut self, pass: Box<dyn FilterPass>) {
        #[cfg(debug_assertions)]
        debug!("Adding pass: {}", pass.name());
        self.passes.push(pass);
    }

    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        if self.shared_buffer.len() < buffer.len() {
            self.shared_buffer.resize(buffer.len(), 0.0);
        }
        self.shared_buffer[..buffer.len()].copy_from_slice(buffer);

        let total_start_time = Instant::now();
        for pass in self.passes.iter_mut() {
            let pass_start_time = Instant::now();
            pass.process(&mut self.shared_buffer[..buffer.len()]);
            debug!(
                "Pass '{}' took: {:?}",
                pass.name(),
                pass_start_time.elapsed()
            );
        }
        buffer.copy_from_slice(&self.shared_buffer[..buffer.len()]);

        debug!("------------------------------------");
        debug!("Total processing time: {:?}", total_start_time.elapsed());
    }
}
