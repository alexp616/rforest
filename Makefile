BUILD_DIR         = build
BENCH_BUILD_DIR   = build_bench
FFT_BENCH_BUILD_DIR = build_fft_bench

.PHONY: all test bench bench_fft clean

all:
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null
	@cmake --build $(BUILD_DIR)

test: all
	@ctest --test-dir $(BUILD_DIR) -V --output-on-failure

bench:
	@cmake -S . -B $(BENCH_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBENCH_PROFILE=ON -Wno-dev > /dev/null
	@cmake --build $(BENCH_BUILD_DIR) --target bench_matmul
	@echo "Running benchmarks..."
	@$(BENCH_BUILD_DIR)/bench_matmul

bench_fft:
	@cmake -S . -B $(FFT_BENCH_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null
	@cmake --build $(FFT_BENCH_BUILD_DIR) --target bench_fft_direct
	@echo "Running direct FFT benchmark..."
	@$(FFT_BENCH_BUILD_DIR)/bench_fft_direct

clean:
	@rm -rf $(BUILD_DIR) $(BENCH_BUILD_DIR) $(FFT_BENCH_BUILD_DIR)
