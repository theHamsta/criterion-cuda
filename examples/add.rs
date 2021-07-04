use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use criterion_cuda::CudaTime;
use std::ffi::CString;

use criterion::BenchmarkId;
use rustacuda::{
    launch, memory::DeviceBuffer, module::Module, stream::Stream, stream::StreamFlags,
};

/// Profiles a simple element-wise addition kernel
pub fn cuda_bench(c: &mut Criterion<CudaTime>) {
    let _ctx = rustacuda::quick_init().expect("could not create CUDA context");

    let module_data =
        CString::new(include_str!("../resources/add.ptx")).expect("Could not read PTX");
    let module = Module::load_from_string(&module_data).expect("Could not load PTX");
    let mut group = c.benchmark_group("add kernel");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create stream");

    const SMALL: u32 = 2_000;
    const BIG: u32 = 20_000;

    unsafe {
        let mut x = DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");
        let mut y = DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");
        let mut result =
            DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");

        for buffer_size in [SMALL, BIG] {
            group.throughput(Throughput::Bytes(buffer_size as u64 * 4));
            group.bench_function(BenchmarkId::new("add kernel", buffer_size), |b| {
                b.iter(|| {
                    launch!(module.sum<<<buffer_size, 1, 0, stream>>>(
                    //// Try this change!
                    //launch!(module.sum<<<256, ((i + 256 - 1) / 256), 0, stream>>>(
                        x.as_device_ptr(),
                        y.as_device_ptr(),
                        result.as_device_ptr(),
                        buffer_size
                    ))
                    .expect("Failed to launch CUDA kernel")
                });
            });
        }
    }

    group.finish()
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(CudaTime);
    targets = cuda_bench
);
criterion_main!(bench);
