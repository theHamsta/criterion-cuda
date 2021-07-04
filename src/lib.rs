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
#[test]
fn init_cuda_test() {
    let _ctx = rustacuda::quick_init().expect("could not create CUDA context");
}

use criterion::{criterion_group, criterion_main, Criterion};
mod bench {
    use std::ffi::CString;

    use criterion::BenchmarkId;
    use rustacuda::{launch, memory::DeviceBuffer, module::Module, stream::StreamFlags};

    use super::*;

    pub fn cuda_bench(c: &mut Criterion<CudaTime>) {
        let _ctx = rustacuda::quick_init().expect("could not create CUDA context");

        let module_data =
            CString::new(include_str!("../resources/add.ptx")).expect("Could not read PTX");
        let module = Module::load_from_string(&module_data).expect("Could not load PTX");
        let mut group = c.benchmark_group("cuda kernel");
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create stream");

        const SMALL: u32 = 2_000;
        const BIG: u32 = 20_000;

        unsafe {
            let mut x =
                DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");
            let mut y =
                DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");
            let mut result =
                DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");

            for i in [SMALL, BIG] {
                group.bench_function(BenchmarkId::new("cuda kernel", i), |b| {
                    b.iter(|| {
                        launch!(module.sum<<<i, 1, 0, stream>>>(
                        //// Try this change!
                        //launch!(module.sum<<<256, ((i + 256 - 1) / 256), 0, stream>>>(
                            x.as_device_ptr(),
                            y.as_device_ptr(),
                            result.as_device_ptr(),
                            i
                        ))
                        .expect("Failed to launch CUDA kernel")
                    });
                });
            }
        }

        group.finish()
    }
}
criterion_group!(
    name = cuda_bench;
    config = Criterion::default().with_measurement(CudaTime);
    targets = bench::cuda_bench
);
criterion_main!(cuda_bench);
