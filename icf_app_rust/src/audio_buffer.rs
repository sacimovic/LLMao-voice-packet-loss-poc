use std::collections::VecDeque;

pub struct AudioBuffer {
    buffer: VecDeque<i16>,
    max_samples: usize,
}

impl AudioBuffer {
    pub fn new(sample_rate: usize, duration_ms: u64) -> Self {
        let max_samples = (sample_rate as u64 * duration_ms / 1000) as usize;
        Self {
            buffer: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    pub fn add_samples(&mut self, samples: &[i16]) {
        for &sample in samples {
            if self.buffer.len() >= self.max_samples {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
    }

    pub fn get_samples(&self) -> Vec<i16> {
        self.buffer.iter().copied().collect()
    }
}

