import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from typing import List


def load_wikitext(max_samples: int = 200, min_chars: int = 50) -> List[str]:
    """Load WikiText-2 validation set, filtering to usable sentences."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", trust_remote_code=False)
    texts = [
        t.strip() for t in dataset["text"]
        if t.strip() and len(t.strip()) >= min_chars
        and not t.strip().startswith("=")
    ]
    return texts[:max_samples]


def tokenize_batch(tokenizer: GPT2Tokenizer, texts: List[str], max_length: int = 1024):
    """Tokenize a batch of texts for GPT-2."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return enc["input_ids"], enc.get("attention_mask", None)
