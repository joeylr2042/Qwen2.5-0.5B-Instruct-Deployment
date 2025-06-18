import json
import logging
import random
from typing import TYPE_CHECKING, List, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# --- logs --- #
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

def get_tokenizer(
        tokenizer_name: str,
        *args,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "PreTrainedTokenizerBase":
    logger.info(f"Loading tokenizer {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, trust_remote_code = True, **kwargs
        )
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            logger.warning(f"Loading tokenizer {tokenizer_name} failed")
        logger.info(f"Tokenizer {tokenizer_name} loaded")
        return tokenizer
    except Exception as e:
        logger.warning(f"Loading tokenizer {tokenizer_name} failed")
        raise e

def sample_requests(
        data_path: str,
        num_requests: int,
        tokenizer: "PreTrainedTokenizerBase"
) -> List[Tuple[str, int, int]]:
    logger.info(f"Sampling {num_requests} requests")
    try:
        with open(data_path, encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error(f"File {data_path} not found")
        return []
    except json.JSONDecodeError:
        logger.error(f"File {data_path} could not be decoded")
        return []

    # Filter out invalid dialogue (less than 2 turns or first turn is not 'human')
    valid_conversations = [
        data for data in dataset
        if len(data.get("conversations", [])) >= 2 and data["conversations"][0].get("from") == "human"
    ]

    # Extract valid prompt-completion pairs
    request_pairs = [
        (conv["conversations"][0]["value"], conv["conversations"][1]["value"])
        for conv in valid_conversations
    ]

    if len(request_pairs) < num_requests:
        logger.warning(
            f"Number of valid conversations in the dataset ({len(request_pairs)}) is less than the number of samples requested ({num_requests})."
            "All available valid conversations will be used."
        )
        sampled_pairs = request_pairs
    else:
        sampled_pairs = random.sample(request_pairs, num_requests)

    prompts = [p for p, c in sampled_pairs]
    completions = [c for p, c in request_pairs]

    logger.info(f"Sampling {len(prompts)} prompts")
    prompt_token_ids = tokenizer(prompts).input_ids
    completion_token_ids = tokenizer(completions).input_ids
    logger.info(f"Tokenization finished")

    final_requests = []
    for i in range(len(sampled_pairs)):
        prompt_len = len(prompt_token_ids[i])
        output_len = len(completion_token_ids[i])
        if prompt_len > 4 and output_len > 4:
            final_requests.append((prompts[i], prompt_len, output_len))

    logger.info(f"Data preparation finished")
    return final_requests