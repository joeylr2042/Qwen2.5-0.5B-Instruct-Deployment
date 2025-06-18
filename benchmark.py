import asyncio
import json
import time
import logging
import random
from typing import List, Tuple

import aiohttp
import numpy as np

from util import get_tokenizer, sample_requests

# --- Logs and environment variables configuration --- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUEST_LATENCY: List[Tuple[int, int, float]] = []

# --- Configuration --- #
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-0.5b-instruct"

# Stress testing parameters
NUM_REQUESTS = 100
CONCURRENCY = 10

# Dataset and tokenizer settings
DATASET_PATH = 'ShareGPT_V3_unfiltered_cleaned_split.json'
TOKENIZER_NAME = "qwen/Qwen2.5-0.5B-Instruct"

async def send_request(session, prompt, prompt_len, output_len):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": output_len + 20,
        "stream": False
    }

    request_start_time = time.time()
    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                request_latency = time.time() - request_start_time
                REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
            else:
                logger.error(f"Request failed with status {response.status}")
    except aiohttp.ClientError as e:
        logger.error(f"Request failed with status {e}")

class BenchmarkRunner:
    """The executor which is responsible for managing concurrency and task queues during press testing."""

    def __init__(self, requests: List[Tuple[str, int, int]], concurrency: int):
        self.requests = requests
        self.request_queue = asyncio.Queue()
        for request in requests:
            self.request_queue.put_nowait(request)
        self.concurrency = concurrency

    async def worker(self, session: aiohttp.ClientSession):
        """A single worker continuously takes tasks from the queue and executes them."""
        while not self.request_queue.empty():
            try:
                prompt, prompt_len, output_len = self.request_queue.get_nowait()
                await send_request(session, prompt, prompt_len, output_len)
                self.request_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def run(self):
        """Start all workers and wait for all tasks to complete."""
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(self.worker(session))
                for _ in range(self.concurrency)
            ]
            await self.request_queue.join()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        return end_time - start_time

def main():
    logger.info("--- LLM API press testing ---")
    tokenizer = get_tokenizer(TOKENIZER_NAME, trust_remote_code=True)
    input_requests = sample_requests(DATASET_PATH, NUM_REQUESTS, tokenizer)

    if not input_requests:
        logger.error("Fail to load any requests. Press testing stopped.")
        return

    logger.info(f"press testing configuration: concurrency={CONCURRENCY} requests={len(input_requests)}")
    logger.info(f"Benchmark start...")

    runner = BenchmarkRunner(input_requests, CONCURRENCY)
    benchmark_start_time = time.time()
    benchmark_time = asyncio.run(runner.run())

    logger.info(f"--- Benchmark result analysis ---")
    if not REQUEST_LATENCY:
        logger.info(f"No any requests were sent during the press testing.")
        return

    total_requests = len(REQUEST_LATENCY)

    print(f"\nTotal time consumption: {benchmark_time: .2f}s")
    print(f"Completed requests: {total_requests}")

    throughput_req_per_s = total_requests / benchmark_time
    print(f"Throughput requests per second: {throughput_req_per_s:.2} s")

    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")

    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
        if (prompt_len + output_len) > 0
    ]) * 1000
    print(f"Average latency per token: {avg_per_token_latency:.2f} ms")

    total_output_tokens = sum([output_len for _, output_len, _ in REQUEST_LATENCY])
    throughput_token_per_s = total_output_tokens / benchmark_time
    print(f"Throughput (TPS): {throughput_token_per_s:.2f} Output Token/second")


if __name__ == "__main__":
    main()