import time
import torch


def benchmark_matmul(
    d: int,
    num_iterations: int,
    warmup_iterations: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Benchmarks matrix multiplication performance on a specified device.

    Args:
        d (int): The dimension of the square matrices.
        num_iterations (int): The number of iterations to run the benchmark for.
        warmup_iterations (int): The number of warmup iterations to run before benchmarking.
        device (str): The device to run the benchmark on (e.g., "cuda", "cpu").
        dtype (torch.dtype): The data type for the tensors (e.g., torch.bfloat16).
    """
    x = torch.randn(size=(d, d), dtype=dtype).to(device)
    y = torch.randn(size=(d, d), dtype=dtype).to(device)

    def matmul_op(tensor):
        # The operation to be benchmarked
        return tensor @ y.T

    # Warmup phase to let the GPU reach a stable state
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        for _ in range(50):
            x = matmul_op(x)
        if device == "cuda":
            torch.cuda.synchronize()

    # Benchmark phase
    print(f"Running {num_iterations} benchmark iterations...")
    tic = time.time()
    for _ in range(num_iterations):
        for _ in range(50):
            x = matmul_op(x)
        if device == "cuda":
            torch.cuda.synchronize()
    toc = time.time()

    # --- Performance Calculation ---
    elapsed_seconds = toc - tic
    milliseconds = 1e3 * elapsed_seconds

    # TFLOPs calculation:
    # (d**3) * 2: operations for one matrix multiplication (d*d*d multiplications and d*d*(d-1) additions ~ 2*d^3)
    # * 50: inner loop repetitions
    # * num_iterations: outer loop repetitions
    # / (1024**4): to convert from operations to Tera-FLOPs
    total_tera_flops = (d**3) * 2 * 50 * num_iterations / (1024**4)
    tflops_per_second = total_tera_flops / elapsed_seconds if elapsed_seconds > 0 else 0

    print("\n--- Benchmark Results ---")
    print(f"Total time: {milliseconds:.3f} ms")
    print(f"Achieved TFLOPs: {tflops_per_second:.3f}")


if __name__ == "__main__":
    # --- Configuration ---
    MATRIX_DIMENSION = 8192
    BENCHMARK_ITERATIONS = 50  # Increase for a longer, more stable benchmark
    WARMUP_ITERATIONS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_TYPE = torch.bfloat16

    print("Starting benchmark with the following configuration:")
    print(f"  Matrix Dimension: {MATRIX_DIMENSION}x{MATRIX_DIMENSION}")
    print(f"  Benchmark Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Warmup Iterations: {WARMUP_ITERATIONS}")
    print(f"  Device: {DEVICE}")
    print(f"  Data Type: {DATA_TYPE}")
    print("-" * 20)

    benchmark_matmul(
        d=MATRIX_DIMENSION,
        num_iterations=BENCHMARK_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS,
        device=DEVICE,
        dtype=DATA_TYPE,
    )
