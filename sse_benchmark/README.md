# SSE Benchmark

A Binary used to help benchmark various aspects of the SSE Solver

## Running

To run a benchmark use

```bash
cargo build --release
./target/release/sse_benchmark
```

To obtain a flamegraph use

```bash
perf record --call-graph dwarf <your application>
hotspot perf.data
```

make sure to install perf and hotspot first

```
sudo apt update && sudo apt upgrade -y && sudo  apt install linux-perf hotspot -y
sudo ln -sf <actual perf in usr/bin/perf_5.X> /usr/bin/perf
```
