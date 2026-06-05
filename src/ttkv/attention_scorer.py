"""Attention-guided salience scorer for KV cache compression."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .core import CacheConfig, TieredKVCache


class AttentionGuidedScorer:

    def __init__(self, ema_decay: float = 0.95, structural_floor: float = 0.1):
        self.ema_decay = ema_decay
        self.structural_floor = structural_floor
        self.position_importance = {}
        self.structural_scores = {}
        self.seen_positions = set()

    def compute_structural_score(self, token_ids: torch.Tensor,
                                 tokenizer=None) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        scores = torch.zeros(batch_size, seq_len)

        for b in range(batch_size):
            for pos in range(seq_len):
                token_id = int(token_ids[b, pos])

                if token_id in self.structural_scores:
                    scores[b, pos] = self.structural_scores[token_id]
                    continue

                score = 0.0
                if pos < seq_len // 20 or pos > seq_len * 19 // 20:
                    score = max(score, 0.6)
                if pos % 10 == 0:
                    score = max(score, 0.4)

                self.structural_scores[token_id] = score
                scores[b, pos] = score

        return scores

    def update_from_attention(self, attention_weights: torch.Tensor,
                            query_position: int,
                            generated_token_id: int):
        if attention_weights.dim() == 2:
            attn = attention_weights.mean(dim=0)
        else:
            attn = attention_weights

        seq_len = attn.size(0)

        for pos in range(seq_len):
            attn_val = float(attn[pos])
            if pos in self.position_importance:
                self.position_importance[pos] = (
                    self.ema_decay * self.position_importance[pos] +
                    (1 - self.ema_decay) * attn_val
                )
            else:
                self.position_importance[pos] = attn_val
            self.seen_positions.add(pos)

    def get_salience_scores(self, seq_len: int,
                           structural_scores: Optional[torch.Tensor] = None
                           ) -> torch.Tensor:
        scores = torch.zeros(seq_len)

        for pos in range(seq_len):
            attn_score = self.position_importance.get(pos, 0.0)
            struct_score = 0.0
            if structural_scores is not None and pos < len(structural_scores[0]):
                struct_score = float(structural_scores[0, pos])
            scores[pos] = max(attn_score, struct_score * self.structural_floor)

        return scores

    def decay_unseen_positions(self, current_seq_len: int):
        positions_to_decay = []
        for pos in list(self.position_importance.keys()):
            if pos >= current_seq_len:
                positions_to_decay.append(pos)

        for pos in positions_to_decay:
            del self.position_importance[pos]
            self.seen_positions.discard(pos)

    def reset(self):
        self.position_importance.clear()
        self.structural_scores.clear()
        self.seen_positions.clear()


class AttentionBasedKVCache:

    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.scorer = AttentionGuidedScorer(
            ema_decay=0.95,
            structural_floor=0.1
        )

    def compress_with_attention(self, k: torch.Tensor, v: torch.Tensor,
                               attention_weights: torch.Tensor,
                               token_ids: torch.Tensor,
                               positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        batch_size, num_heads, seq_len, head_dim = k.shape

        self.scorer.update_from_attention(
            attention_weights[0],
            query_position=seq_len - 1,
            generated_token_id=int(token_ids[0, -1]) if token_ids.size(1) > 0 else 0
        )

        structural_scores = self.scorer.compute_structural_score(token_ids)
        salience = self.scorer.get_salience_scores(seq_len, structural_scores)
        salience = salience.unsqueeze(0).repeat(batch_size, 1)

        cache = TieredKVCache(self.config)
        cache.add(k, v, salience, positions)

        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        if k_comp is not None:
            self.scorer.decay_unseen_positions(k_comp.size(2))

        return k_comp, v_comp, stats


def extract_attention_weights(model_outputs) -> Optional[torch.Tensor]:
    if not hasattr(model_outputs, 'attentions'):
        return None

    attentions = model_outputs.attentions
    if attentions is None or len(attentions) == 0:
        return None

    last_layer_attn = attentions[-1]
    last_pos_attn = last_layer_attn[:, :, -1, :]

    return last_pos_attn


class AttentionGuidedWrapper:

    def __init__(self, model, tokenizer, cache_config):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_config = cache_config
        self.attention_cache = AttentionBasedKVCache(cache_config, tokenizer)

    def generate_with_attention_guidance(self, input_ids: torch.Tensor,
                                        max_new_tokens: int = 50,
                                        temperature: float = 0.7) -> Tuple[str, List[Dict]]:
        if hasattr(self.model, 'device'):
            input_ids = input_ids.to(self.model.device)

        generated = input_ids
        all_stats = []
        compressed_past = None

        for step in range(max_new_tokens):
            with torch.no_grad():
                if compressed_past is not None:
                    # NOTE: Shape mismatches between compressed cache and model
                    # expectations (e.g. with rotary embeddings or GQA) may require
                    # model-specific patching for full integration.
                    outputs = self.model(
                        generated[:, -1:],
                        past_key_values=compressed_past,
                        use_cache=False,
                        output_attentions=True
                    )
                else:
                    outputs = self.model(
                        generated,
                        use_cache=True,
                        output_attentions=True
                    )

            attn_weights = extract_attention_weights(outputs)
            past_kv = compressed_past if compressed_past is not None else outputs.past_key_values

            if attn_weights is not None and past_kv is not None:
                compressed_past_list = []
                for layer_idx, layer_cache in enumerate(past_kv):
                    k = layer_cache[0]
                    v = layer_cache[1]

                    seq_len = k.size(2)
                    positions = torch.arange(seq_len, device=k.device).unsqueeze(0)
                    token_ids = generated[:, :seq_len]

                    k_comp, v_comp, stats = self.attention_cache.compress_with_attention(
                        k, v, attn_weights, token_ids, positions
                    )

                    compressed_past_list.append((k_comp, v_comp))

                    if layer_idx == 0:
                        all_stats.append(stats)

                compressed_past = tuple(compressed_past_list)

            logits = outputs.logits
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return result, all_stats
