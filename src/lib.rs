use criterion::{
    measurement::{Measurement, ValueFormatter},
    Throughput,
};
use rustacuda::prelude::Stream;

/// `CudaTime` measures the time of one or multiple CUDA kernels via CUDA events
pub struct CudaTime;

impl Measurement for CudaTime {
    type Intermediate = rustacuda::event::Event;
    type Value = f32;

    fn start(&self) -> Self::Intermediate {
        let event = rustacuda::event::Event::new(rustacuda::event::EventFlags::DEFAULT)
            .expect("Failed to create event");
        event
            .record(&Stream::null())
            .expect("could not record stream");
        event
    }

    fn end(&self, start_event: Self::Intermediate) -> Self::Value {
        let end_event = rustacuda::event::Event::new(rustacuda::event::EventFlags::DEFAULT)
            .expect("Failed to create event");
        end_event
            .record(&Stream::null())
            .expect("could not record stream");
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
            Throughput::Bytes(b) => format!("{:.4} ms per byte", value / *b as f64),
            Throughput::Elements(b) => format!("{:.4} cycles/{}", value, b),
        }
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        " ms"
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        match throughput {
            Throughput::Bytes(n) => {
                for val in values {
                    *val /= *n as f64;
                }
                "cpb"
            }
            Throughput::Elements(n) => {
                for val in values {
                    *val /= *n as f64;
                }
                "c/e"
            }
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "cycles"
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn init_cuda_test() {
        let _ctx = rustacuda::quick_init().expect("could not create CUDA context");
    }
}
