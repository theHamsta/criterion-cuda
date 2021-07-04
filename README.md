# criterion-cuda

This crate provides the `Measurement` `CudaTime` for bench marking CUDA kernels using
[criterion-rs](https://github.com/bheisler/criterion.rs).

See [example/add.rs](example/add.rs) for a usage example.

# Running the Example Benchmark

```bash
cargo bench
```

or with [cargo-criterion](https://github.com/bheisler/cargo-criterion) installed

```bash
cargo criterion
```
