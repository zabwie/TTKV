"""Model-agnostic generator for TTKV KV cache compression testing.

Works with any HuggingFace model that supports past_key_values.
Auto-detects model dimensions from config.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ttkv import CacheConfig, TieredKVCache, compute_type_prior_retention


class CompressedGenerator:

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name

        config = AutoConfig.from_pretrained(model_name)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.max_position = getattr(config, "max_position_embeddings", 2048)
        self.rope = hasattr(config, "rope_scaling") or "rope" in str(config.position_embedding_type).lower() if hasattr(config, "position_embedding_type") else True

        print(f"Loaded {model_name}: {self.num_layers}L, {self.hidden_dim}d, "
              f"{self.num_heads}h ({self.num_kv_heads}kv), head_dim={self.head_dim}, "
              f"max_pos={self.max_position}, RoPE={'yes' if self.rope else 'no'}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self._original_seq_len = None

    def _build_cache_config(self, tau: float):
        return CacheConfig(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            tau_threshold=tau,
        )

    def prefill_and_compress(self, input_ids, tau: float = 0.9):
        self._original_seq_len = int(input_ids.shape[1])

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        cache = outputs.past_key_values

        retention = compute_type_prior_retention(input_ids)
        total_before, total_after = 0, 0

        device = cache.layers[0].keys.device
        orig_dtype = cache.layers[0].keys.dtype

        for layer in cache.layers:
            k = layer.keys.detach()
            v = layer.values.detach()
            seq_len = int(k.shape[2])
            total_before += seq_len

            config = self._build_cache_config(tau)
            ttkv_cache = TieredKVCache(config)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            ret = retention[:, :seq_len].to(device)
            if ret.shape[1] < seq_len:
                pad = torch.full((1, seq_len - ret.shape[1]), 0.3, device=device, dtype=ret.dtype)
                ret = torch.cat([ret, pad], dim=1)

            ttkv_cache.add(k, v, ret, positions)
            k_comp, v_comp, _pos_comp = ttkv_cache.get_compressed_cache()
            if k_comp.shape[2] == 0:
                k_comp = k[:, :, :1, :]
                v_comp = v[:, :, :1, :]

            k_comp = k_comp.to(dtype=orig_dtype)
            v_comp = v_comp.to(dtype=orig_dtype)

            total_after += int(k_comp.shape[2])
            layer.keys = k_comp.clone()
            layer.values = v_comp.clone()

        return cache, {
            "compression_ratio": total_before / max(total_after, 1),
            "tokens_before": total_before,
            "tokens_after": total_after,
        }

    def measure_memory(self, input_ids, tau: float = 0.9):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        cache = outputs.past_key_values
        torch.cuda.synchronize()

        uncomp_bytes = sum(
            l.keys.element_size() * l.keys.numel()
            + l.values.element_size() * l.values.numel()
            for l in cache.layers
        )

        retention = compute_type_prior_retention(input_ids)
        device = cache.layers[0].keys.device
        orig_dtype = cache.layers[0].keys.dtype
        total_before, total_after = 0, 0

        for layer in cache.layers:
            k = layer.keys.detach()
            v = layer.values.detach()
            seq_len = int(k.shape[2])
            total_before += seq_len

            config = self._build_cache_config(tau)
            ttkv_cache = TieredKVCache(config)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            ret = retention[:, :seq_len].to(device)
            if ret.shape[1] < seq_len:
                pad = torch.full((1, seq_len - ret.shape[1]), 0.3, device=device, dtype=ret.dtype)
                ret = torch.cat([ret, pad], dim=1)

            ttkv_cache.add(k, v, ret, positions)
            k_comp, v_comp, _pos_comp = ttkv_cache.get_compressed_cache()
            if k_comp.shape[2] == 0:
                k_comp = k[:, :, :1, :]
                v_comp = v[:, :, :1, :]

            k_comp = k_comp.to(dtype=orig_dtype)
            v_comp = v_comp.to(dtype=orig_dtype)
            total_after += int(k_comp.shape[2])
            layer.keys = k_comp
            layer.values = v_comp

        comp_bytes = sum(
            l.keys.element_size() * l.keys.numel()
            + l.values.element_size() * l.values.numel()
            for l in cache.layers
        )

        return {
            "seq_len": int(input_ids.shape[1]),
            "kv_uncompressed_mb": round(uncomp_bytes / 1024 / 1024, 2),
            "kv_compressed_mb": round(comp_bytes / 1024 / 1024, 2),
            "compression_ratio": round(uncomp_bytes / max(comp_bytes, 1), 2),
            "memory_saved_pct": round((1 - comp_bytes / max(uncomp_bytes, 1)) * 100, 1),
            "tokens_before": total_before,
            "tokens_after": total_after,
            "model": self.model_name,
        }
