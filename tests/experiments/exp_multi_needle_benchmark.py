#!/usr/bin/env python3
import torch
import json
import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ttkv import auto_config, build_retention_mask, CacheConfig, TieredKVCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LENGTH = 3600
MAX_NEW_TOKENS = 40
TTKV_TARGET_SIGNAL_PCT = 3.5

NEEDLES = [
    {
        "type": "password",
        "needle": "The database password is ALPHA-77-BRAVO and must be rotated monthly.",
        "query": "What is the database password?",
        "key_facts": ["ALPHA-77-BRAVO"],
    },
    {
        "type": "password",
        "needle": "Access code: 8F3A-1B2C is required for the secure vault.",
        "query": "What is the access code for the secure vault?",
        "key_facts": ["8F3A-1B2C"],
    },
    {
        "type": "password",
        "needle": "The encryption key for sector seven is KAPPA-99-ZULU.",
        "query": "What is the encryption key for sector seven?",
        "key_facts": ["KAPPA-99-ZULU"],
    },
    {
        "type": "name",
        "needle": "The CEO is Dr. Maria Santos from Stanford University.",
        "query": "Who is the CEO and where are they from?",
        "key_facts": ["Maria Santos", "Stanford"],
    },
    {
        "type": "name",
        "needle": "Ambassador James Whitfield signed the treaty on behalf of the coalition.",
        "query": "Which ambassador signed the treaty?",
        "key_facts": ["James Whitfield"],
    },
    {
        "type": "number",
        "needle": "Revenue reached 3.2 billion dollars in fiscal year 2024.",
        "query": "What was the revenue reported for fiscal year 2024?",
        "key_facts": ["3.2 billion"],
    },
    {
        "type": "number",
        "needle": "The temperature record is minus 89.2 degrees Celsius at Vostok Station.",
        "query": "What is the temperature record at Vostok Station?",
        "key_facts": ["89.2"],
    },
    {
        "type": "number",
        "needle": "The company has exactly 1,847 employees across twelve offices worldwide.",
        "query": "How many employees does the company have?",
        "key_facts": ["1,847", "1847"],
    },
    {
        "type": "location",
        "needle": "The server is located at 742 Evergreen Terrace, Springfield.",
        "query": "Where is the server located?",
        "key_facts": ["Evergreen Terrace", "Springfield"],
    },
    {
        "type": "location",
        "needle": "Meeting at the Crystal Palace Hotel, room 401, at noon.",
        "query": "Where is the meeting and in which room?",
        "key_facts": ["Crystal Palace", "401"],
    },
    {
        "type": "technical",
        "needle": "The API endpoint is https://api.example.com/v2/authenticate for all services.",
        "query": "What is the API endpoint for authentication?",
        "key_facts": ["api.example.com"],
    },
    {
        "type": "technical",
        "needle": "Error code E-4051 indicates a critical authentication failure.",
        "query": "What does error code E-4051 indicate?",
        "key_facts": ["authentication failure", "E-4051"],
    },
    {
        "type": "mixed",
        "needle": "Patient number 8842 was prescribed 50 milligrams of Cefalexin by Dr. Nakamura.",
        "query": "What medication was prescribed to Patient 8842 and by whom?",
        "key_facts": ["Cefalexin", "Nakamura"],
    },
    {
        "type": "mixed",
        "needle": "Flight AZ dash 774 departs from Gate B dash 12 at 14 30 to Tokyo Haneda.",
        "query": "What is the flight number, gate, and destination?",
        "key_facts": ["774", "B-12", "Tokyo"],
    },
]


def load_filler_text(target_words: int = 700) -> str:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                           trust_remote_code=False)
    parts = []
    wc = 0
    for item in dataset["text"]:
        t = item.strip()
        if not t or t.startswith("="):
            continue
        parts.append(t)
        wc += len(t.split())
        if wc >= target_words:
            break
    return " ".join(parts)


class BenchmarkRunner:

    def __init__(self):
        print(f"Loading {MODEL_NAME} on {DEVICE}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
            attn_implementation="eager")
        self.model.eval()

        config = AutoConfig.from_pretrained(MODEL_NAME)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        print(f"  Model: {self.num_layers}L, {self.hidden_dim}d, "
              f"{self.num_heads}h ({self.num_kv_heads}kv), head_dim={self.head_dim}")

        self.filler = load_filler_text(target_words=3000)
        print(f"  Filler: {len(self.filler.split())} words loaded")

    def build_prompt(self, entry: dict, target_tokens: int = CONTEXT_LENGTH
                     ) -> Tuple[str, str, torch.Tensor, int, int]:
        needle_text = entry["needle"]
        query = entry["query"]
        needle_ids = self.tokenizer.encode(needle_text, add_special_tokens=False)
        query_str = f"\n\nQuestion: {query}\nAnswer:"
        query_ids = self.tokenizer.encode(query_str, add_special_tokens=False)
        reserved = len(needle_ids) + len(query_ids) + 10

        filler_ids = self.tokenizer.encode(self.filler, add_special_tokens=False)
        max_filler = max(0, target_tokens - reserved)
        if len(filler_ids) > max_filler:
            filler_ids = filler_ids[:max_filler]
        filler_text = self.tokenizer.decode(filler_ids, skip_special_tokens=True)

        prompt = f"{needle_text}\n\n{filler_text}{query_str}"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

        if input_ids.shape[1] > target_tokens:
            input_ids = input_ids[:, :target_tokens]

        actual_len = input_ids.shape[1]

        n_start, n_end = 0, min(len(needle_ids), actual_len)
        seq = input_ids[0].tolist()
        for i in range(len(seq) - len(needle_ids) + 1):
            if seq[i:i + len(needle_ids)] == needle_ids:
                n_start, n_end = i, i + len(needle_ids)
                break

        return prompt, query, input_ids, n_start, n_end

    def prefill(self, input_ids, output_attentions=False):
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True,
                                 output_attentions=output_attentions)
        return outputs.past_key_values, (
            outputs.attentions if output_attentions else None)

    def compress_h2o(self, past_key_values, attentions, budget):
        last_attn = attentions[-1]
        attn_seq = last_attn.shape[2]
        scores = last_attn[0, :, -1, :].mean(dim=0)

        if budget >= attn_seq:
            total = sum(int(l.keys.shape[2]) for l in past_key_values.layers)
            return past_key_values, total, total

        _, top = torch.topk(scores, budget)
        top = top.sort()[0]

        tb, ta = 0, 0
        for layer in past_key_values.layers:
            k, v = layer.keys.detach(), layer.values.detach()
            ls = int(k.shape[2])
            tb += ls
            valid_top = top[top < ls]
            if len(valid_top) == 0:
                valid_top = torch.arange(min(budget, ls), device=top.device)
            layer.keys = k[:, :, valid_top, :].clone()
            layer.values = v[:, :, valid_top, :].clone()
            ta += int(layer.keys.shape[2])
        return past_key_values, tb, ta

    def compress_ttkv(self, past_key_values, n_start, n_end, input_ids):
        seq_len = int(input_ids.shape[1])
        device = input_ids.device
        retention = build_retention_mask(
            n_tokens=seq_len, needle_ranges=[(n_start, n_end)],
            recent_window=16, base_score=0.3, protect_score=1.0).to(device)

        config = auto_config(n_tokens=seq_len,
                             protected_indices=(retention > 0.9)[0],
                             target_signal_pct=TTKV_TARGET_SIGNAL_PCT)
        config.num_heads = self.num_kv_heads
        config.head_dim = self.head_dim
        config.hidden_dim = self.hidden_dim

        tb, ta = 0, 0
        for layer in past_key_values.layers:
            k, v = layer.keys.detach(), layer.values.detach()
            ls = int(k.shape[2])
            tb += ls

            cache = TieredKVCache(config)
            pos = torch.arange(ls, device=device).unsqueeze(0)
            ret = retention[:, :ls].to(device)
            if ret.shape[1] < ls:
                pad = torch.full((1, ls - ret.shape[1]), 0.3, device=device, dtype=ret.dtype)
                ret = torch.cat([ret, pad], dim=1)

            cache.add(k, v, ret, pos)
            kc, vc, _ = cache.get_compressed_cache()
            if kc.shape[2] == 0:
                kc, vc = k[:, :, :1, :], v[:, :, :1, :]

            ta += int(kc.shape[2])
            layer.keys = kc.to(dtype=k.dtype).clone()
            layer.values = vc.to(dtype=v.dtype).clone()

        return past_key_values, tb / max(ta, 1), tb, ta

    def generate(self, past_key_values, orig_seq_len, input_ids,
                 max_new=MAX_NEW_TOKENS):
        device = past_key_values.layers[0].device
        gids = []
        last_token = input_ids[:, -1:]
        pkv = past_key_values

        for step in range(max_new):
            pos = orig_seq_len + step
            pids = torch.tensor([[pos]], device=device, dtype=torch.long)
            with torch.no_grad():
                out = self.model(input_ids=last_token, past_key_values=pkv,
                                 position_ids=pids, use_cache=True)
            pkv = out.past_key_values
            nt = out.logits[:, -1, :].argmax(dim=-1).item()
            if nt == self.tokenizer.eos_token_id:
                break
            gids.append(nt)
            last_token = torch.tensor([[nt]], device=device, dtype=torch.long)

        return self.tokenizer.decode(gids, skip_special_tokens=True)

    def check_recall(self, output, key_facts):
        ol = output.lower().strip()
        return any(f.lower() in ol for f in key_facts)

    def run_single_test(self, entry, method, budget=None):
        _, _, input_ids, ns, ne = self.build_prompt(entry)
        seq_len = int(input_ids.shape[1])
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            past_kv, attn = self.prefill(input_ids, output_attentions=(method == "h2o"))
            tb = sum(int(l.keys.shape[2]) for l in past_kv.layers)
            ta = tb
            ratio = 1.0

            if method == "h2o":
                past_kv, tb, ta = self.compress_h2o(past_kv, attn, budget)
                ratio = tb / max(ta, 1)
            elif method == "ttkv":
                past_kv, ratio, tb, ta = self.compress_ttkv(past_kv, ns, ne, input_ids)

            output = self.generate(past_kv, seq_len, input_ids)
            recall = self.check_recall(output, entry["key_facts"])
            return {"recall": recall, "ratio": round(ratio, 2),
                    "output": output.strip(), "tokens_before": tb,
                    "tokens_after": ta, "error": None}
        except Exception as e:
            return {"recall": False, "ratio": 0.0, "output": "",
                    "tokens_before": 0, "tokens_after": 0, "error": str(e)}

    def run_full_benchmark(self):
        print(f"\n{'='*80}")
        print(f"MULTI-NEEDLE SLOW-BURN BENCHMARK")
        print(f"Model: {MODEL_NAME}  |  Context: {CONTEXT_LENGTH} tokens")
        print(f"Needles: {len(NEEDLES)}  |  Methods: H2O(1024,768) + TTKV(auto)")
        print(f"{'='*80}\n")

        results = []
        methods = [("h2o_1024", "h2o", 1024), ("h2o_768", "h2o", 768),
                   ("ttkv_auto", "ttkv", None)]

        for i, entry in enumerate(NEEDLES):
            print(f"[{i+1}/{len(NEEDLES)}] {entry['type']}: {entry['needle'][:60]}...")
            result = {"needle_type": entry["type"], "needle": entry["needle"],
                      "query": entry["query"]}

            for mkey, mname, budget in methods:
                t0 = time.time()
                r = self.run_single_test(entry, mname, budget)
                dt = time.time() - t0
                result[mkey] = {"recall": r["recall"], "ratio": r["ratio"],
                                "output": r["output"],
                                "tokens_after": r["tokens_after"],
                                "error": r["error"]}
                status = "PASS" if r["recall"] else "FAIL"
                err = f" ERR:{r['error'][:40]}" if r["error"] else ""
                print(f"    {mkey:>12}: {status} ratio={r['ratio']:.1f}x "
                      f"time={dt:.1f}s{err}")

            results.append(result)
            torch.cuda.empty_cache()

        summary = {}
        for mkey, _, _ in methods:
            recall_n = sum(1 for r in results if r[mkey]["recall"])
            ratios = [r[mkey]["ratio"] for r in results if r[mkey]["ratio"] > 0]
            avg_r = round(np.mean(ratios), 1) if ratios else 0.0
            summary[f"{mkey}_recall"] = f"{recall_n}/{len(results)}"
            summary[f"{mkey}_avg_ratio"] = avg_r

        ta_list = [r.get("ttkv_auto", {}).get("tokens_after", 0) for r in results]
        summary["avg_ttkv_compressed_total"] = (
            int(np.mean([x for x in ta_list if x > 0])) if ta_list else 0)
        summary["avg_ttkv_ratio"] = summary.get("ttkv_auto_avg_ratio", 0.0)

        output = {"model": MODEL_NAME, "context_length": CONTEXT_LENGTH,
                  "num_needles": len(results), "results": results,
                  "summary": summary}

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  H2O budget 1024:  {summary['h2o_1024_recall']}")
        print(f"  H2O budget 768:   {summary['h2o_768_recall']}")
        print(f"  TTKV auto:         {summary['ttkv_auto_recall']}")
        print(f"  Avg TTKV ratio:    {summary['avg_ttkv_ratio']:.1f}x")
        print(f"{'='*60}")

        return output

    def save_results(self, results, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "..",
                                "results", "multi_needle_benchmark.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {os.path.abspath(path)}")


def main():
    runner = BenchmarkRunner()
    results = runner.run_full_benchmark()
    runner.save_results(results)
    print("\nMulti-needle benchmark complete.")


if __name__ == "__main__":
    main()
