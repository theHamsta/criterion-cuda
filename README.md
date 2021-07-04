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

This `cargo bench` should print
```
add kernel/add kernel/2000
                        time:   [0.0142 ms 0.0142 ms 0.0142 ms]
                        thrpt:  [0.5229 GiB/s 0.5231 GiB/s 0.5232 GiB/s]
                 change:
                        time:   [-1.8762% -1.2732% -0.7326%] (p = 0.00 < 0.05)
                        thrpt:  [+0.7380% +1.2896% +1.9121%]
                        Change within noise threshold.
Found 12 outliers among 100 measurements (12.00%)
  1 (1.00%) low severe
  3 (3.00%) high mild
  8 (8.00%) high severe
add kernel/add kernel/20000
                        time:   [0.1163 ms 0.1163 ms 0.1164 ms]
                        thrpt:  [0.6403 GiB/s 0.6404 GiB/s 0.6404 GiB/s]
                 change:
                        time:   [-1.5252% -1.0335% -0.4522%] (p = 0.00 < 0.05)
                        thrpt:  [+0.4542% +1.0443% +1.5488%]
                        Change within noise threshold.
Found 15 outliers among 100 measurements (15.00%)
  2 (2.00%) low severe
  4 (4.00%) low mild
  5 (5.00%) high mild
  4 (4.00%) high severe
```

## Making an optimization

Now change the following line in the example

```diff
- launch!(module.sum<<<i, 1, 0, stream>>>(
+ launch!(module.sum<<<256, ((i + 256 - 1) / 256), 0, stream>>>(
```

Now the benchmark should run faster:
```
add kernel/add kernel/2000
                        time:   [0.0041 ms 0.0041 ms 0.0041 ms]
                        thrpt:  [1.8300 GiB/s 1.8311 GiB/s 1.8321 GiB/s]
                 change:
                        time:   [-71.520% -71.397% -71.249%] (p = 0.00 < 0.05)
                        thrpt:  [+247.81% +249.61% +251.13%]
                        Performance has improved.
Found 14 outliers among 100 measurements (14.00%)
  4 (4.00%) high mild
  10 (10.00%) high severe
add kernel/add kernel/20000
                        time:   [0.0041 ms 0.0041 ms 0.0041 ms]
                        thrpt:  [18.0229 GiB/s 18.0325 GiB/s 18.0405 GiB/s]
                 change:
                        time:   [-96.459% -96.441% -96.421%] (p = 0.00 < 0.05)
                        thrpt:  [+2694.2% +2709.4% +2724.3%]
                        Performance has improved.
Found 12 outliers among 100 measurements (12.00%)
  4 (4.00%) high mild
  8 (8.00%) high severe
```
