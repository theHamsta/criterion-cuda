use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use criterion_cuda::CudaTime;

use criterion::BenchmarkId;
use cust::{launch, memory::DeviceBuffer, module::Module};

/// Profiles a simple element-wise addition kernel
pub fn cuda_bench(c: &mut Criterion<CudaTime>) {
    let _ctx = cust::quick_init().expect("could not create CUDA context");

    let module_data = include_str!("../resources/add.ptx");
    let module = Module::from_ptx(&module_data, &[]).expect("Could not load PTX");
    let mut group = c.benchmark_group("add kernel");
    let stream = &criterion_cuda::MEASUREMENT_STREAM;

    const SMALL: u32 = 2_000;
    const BIG: u32 = 20_000;

    unsafe {
        let x = DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");
        let y = DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");
        let result = DeviceBuffer::<f32>::zeroed(BIG as usize).expect("Failed to allocate buffer");

        for buffer_size in [SMALL, BIG] {
            group.throughput(Throughput::Bytes(buffer_size as u64 * 4));
            group.bench_function(BenchmarkId::new("add kernel", buffer_size), |b| {
                b.iter(|| {
                    //launch!(module.sum<<<buffer_size, 1, 0, stream>>>(
                    //// Try this change!
                    launch!(module.sum<<<256, ((buffer_size + 256 - 1) / 256), 0, stream>>>(
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
