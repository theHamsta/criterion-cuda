#![doc = include_str!("../README.md")]

use criterion::{
    measurement::{Measurement, ValueFormatter},
    Throughput,
};
use cust::prelude::{Stream, StreamFlags};
use once_cell::sync::Lazy;

/// `CudaTime` measures the time of one or multiple CUDA kernels via CUDA events
pub struct CudaTime;

pub static MEASUREMENT_STREAM: Lazy<Stream> =
    Lazy::new(|| Stream::new(StreamFlags::DEFAULT, None).unwrap());

impl Measurement for CudaTime {
    type Intermediate = cust::event::Event;
    type Value = f32;

    fn start(&self) -> Self::Intermediate {
        let event = cust::event::Event::new(cust::event::EventFlags::DEFAULT)
            .expect("Failed to create event");
        event
            .record(&MEASUREMENT_STREAM)
            .expect("Could not record CUDA event");
        event
    }

    fn end(&self, start_event: Self::Intermediate) -> Self::Value {
        let end_event = cust::event::Event::new(cust::event::EventFlags::DEFAULT)
            .expect("Failed to create event");
        end_event
            .record(&MEASUREMENT_STREAM)
            .expect("Could not record CUDA event");
        end_event.synchronize().expect("Failed to synchronize");
        end_event
            .elapsed_time_f32(&start_event)
            .expect("Failed to measure time")
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1 + v2
    }

    fn zero(&self) -> Self::Value {
        0f32
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &CudaTimeFormatter
    }
}

struct CudaTimeFormatter;

impl ValueFormatter for CudaTimeFormatter {
    fn format_value(&self, value: f64) -> String {
        format!("{:.4} ms", value)
    }

    fn format_throughput(&self, throughput: &Throughput, value: f64) -> String {
        match throughput {
            Throughput::Bytes(b) => format!(
                "{:.4} GiB/s",
                (*b as f64) / (1024.0 * 1024.0 * 1024.0) / (value * 1e-3)
            ),
            Throughput::Elements(b) => format!("{:.4} elements/s", (*b as f64) / (value * 1e-3)),
        }
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "ms"
    }

    /// TODO!
    fn scale_throughputs(
        &self,
        _typical_value: f64,
        throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        match throughput {
            Throughput::Bytes(_) => "GiB/s",
            Throughput::Elements(_) => "elements/s",
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "ms"
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn init_cuda_test() {
        let _ctx = cust::quick_init().expect("could not create CUDA context");
    }
}
