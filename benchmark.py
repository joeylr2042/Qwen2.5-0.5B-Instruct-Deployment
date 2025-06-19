import asyncio
import time
import logging
from typing import List, Tuple

import aiohttp
from util import get_tokenizer, sample_requests
from metrics import calculate_and_print_metrics

# --- Logs and global variables configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

# --- Configuration ---
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-0.5b-instruct"
NUM_REQUESTS = 100
CONCURRENCY = 10
DATASET_PATH = 'ShareGPT_V3_unfiltered_cleaned_split.json'
TOKENIZER_NAME = "qwen/Qwen2.5-0.5B-Instruct"


async def send_request(session, prompt, prompt_len, output_len):
    """Sends a single API request and records the latency."""
    payload = {
        'model': MODEL_NAME, 'messages': [{"role": "user", "content": prompt}],
        "temperature": 0.0, "top_p": 1.0, 'max_tokens': output_len + 20, "stream": False
    }
    request_start_time = time.time()
    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                request_latency = time.time() - request_start_time
                REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
            else:
                logger.error(f"Request failed with status code: {response.status}")
    except aiohttp.ClientError as e:
        logger.error(f"A client error occurred: {e}")


class BenchmarkRunner:
    """Manages the concurrency and task queue for the benchmark."""

    def __init__(self, requests: List[Tuple[str, int, int]], concurrency: int):
        self.requests = requests
        self.request_queue = asyncio.Queue()
        for request in requests: self.request_queue.put_nowait(request)
        self.concurrency = concurrency

    async def worker(self, session: aiohttp.ClientSession):
        """A single worker that fetches and processes requests from the queue."""
        while not self.request_queue.empty():
            try:
                prompt, prompt_len, output_len = self.request_queue.get_nowait()
                await send_request(session, prompt, prompt_len, output_len)
                self.request_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def run(self):
        """Starts the workers and waits for all requests to be processed."""
        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(self.worker(session)) for _ in range(self.concurrency)]
            await self.request_queue.join()
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main():
    """Main function to run the entire benchmark process."""
    logger.info("--- LLM API Stress Test ---")
    tokenizer = get_tokenizer(TOKENIZER_NAME, trust_remote_code=True)
    input_requests = sample_requests(DATASET_PATH, NUM_REQUESTS, tokenizer)

    if not input_requests:
        logger.error("Failed to load any requests. Stopping benchmark.")
        return

    logger.info(f"Benchmark configuration: concurrency={CONCURRENCY}, total requests={len(input_requests)}")
    logger.info("Starting benchmark...")

    runner = BenchmarkRunner(input_requests, CONCURRENCY)

    benchmark_start_time = time.time()
    asyncio.run(runner.run())
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    calculate_and_print_metrics(REQUEST_LATENCY, benchmark_time)


if __name__ == "__main__":
    main()