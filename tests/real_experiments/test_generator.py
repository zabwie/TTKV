import torch
import pytest
from .generator import GPT2Generator


@pytest.fixture(scope="module")
def generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GPT2Generator("gpt2", device=device)


class TestGPT2Generator:
    def test_model_loads(self, generator):
        assert generator.model is not None
        assert generator.tokenizer is not None

    def test_generates_text(self, generator):
        text = "The capital of France is"
        tokens = generator.tokenizer.encode(text, return_tensors="pt").to(generator.device)
        with torch.no_grad():
            outputs = generator.model(tokens, use_cache=True)
        assert outputs.past_key_values is not None
        assert len(outputs.past_key_values.layers) == 12

    def test_compression_reduces_kv(self, generator):
        text = "The quick brown fox jumps over the lazy dog " * 50
        tokens = generator.tokenizer.encode(text, return_tensors="pt").to(generator.device)
        compressed_kv, stats = generator.prefill_and_compress(tokens, tau=0.9)
        assert stats["compression_ratio"] > 1.0, "Compression should reduce tokens"
        assert len(compressed_kv.layers) == 12

    def test_forward_with_compressed_works(self, generator):
        text = "The cat sat on the mat and"
        tokens = generator.tokenizer.encode(text, return_tensors="pt").to(generator.device)
        compressed_kv, _ = generator.prefill_and_compress(tokens, tau=0.9)
        next_token = tokens[:, -1:]
        logits = generator.forward_with_compressed_cache(next_token, compressed_kv)
        assert logits.shape[2] == generator.tokenizer.vocab_size

    def test_perplexity_sanity(self, generator):
        text = "The capital of France is a beautiful city known for its history"
        result = generator.compute_perplexity(text, tau=0.9)
        assert result["ppl_uncompressed"] > 1.0
        assert "compression_ratio" in result
        assert abs(result["ppl_delta_pct"]) < 50.0
