# sse_solver

cargo build --release
./target/release/sse_benchmark 

perf record --call-graph dwarf <your application>
hotspot perf.data

make sure to sudo ln -sf <actual perf in usr/bin/perf_5.X> /usr/bin/perf