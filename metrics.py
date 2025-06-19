import numpy as np
import logging
from typing import List, Tuple

# Get the logger
logger = logging.getLogger(__name__)


def calculate_and_print_metrics(
        request_latency_data: List[Tuple[int, int, float]],
        total_benchmark_time: float
):
    """
    Calculates and prints all performance metrics based on collected latency data and total benchmark time.

    Args:
        request_latency_data: A list of tuples containing latency data, formatted as [(prompt_len, output_len, latency_seconds), ...].
        total_benchmark_time: The total duration of the benchmark in seconds.
    """
    logger.info("--- Benchmark Result Analysis ---")

    if not request_latency_data:
        logger.error("No requests succeeded, cannot calculate metrics.")
        return

    total_requests = len(request_latency_data)

    print(f"\nTotal time taken: {total_benchmark_time:.2f} s")
    print(f"Completed requests: {total_requests}")

    # Throughput (Requests/Second)
    if total_benchmark_time > 0:
        throughput_req_per_s = total_requests / total_benchmark_time
        print(f"Throughput (RPS): {throughput_req_per_s:.2f} req/s")
    else:
        print("Throughput (RPS): N/A (benchmark duration too short)")

    # Latency
    latencies = [latency for _, _, latency in request_latency_data]
    avg_latency = np.mean(latencies)
    print(f"Average latency: {avg_latency:.2f} s")

    # Latency per Token
    valid_token_latencies = [
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in request_latency_data
        if (prompt_len + output_len) > 0
    ]
    if valid_token_latencies:
        avg_per_token_latency = np.mean(valid_token_latencies) * 1000  # Convert to milliseconds
        print(f"Average latency per token: {avg_per_token_latency:.2f} ms")

    # Latency per Output Token
    valid_output_token_latencies = [
        latency / output_len
        for _, output_len, latency in request_latency_data
        if output_len > 0
    ]
    if valid_output_token_latencies:
        avg_per_output_token_latency = np.mean(valid_output_token_latencies) * 1000  # Convert to milliseconds
        print(f"Average latency per output token: {avg_per_output_token_latency:.2f} ms")

    # Output Token Throughput (Tokens/Second)
    if total_benchmark_time > 0:
        total_output_tokens = sum([output_len for _, output_len, _ in request_latency_data])
        throughput_token_per_s = total_output_tokens / total_benchmark_time
        print(f"Throughput (TPS): {throughput_token_per_s:.2f} tokens/s")
    else:
        print("Throughput (TPS): N/A (benchmark duration too short)")
