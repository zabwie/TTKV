import json, os, torch, numpy as np
from tqdm import tqdm
from data_utils import load_wikitext
from generator import GPT2Generator


def build_long_contexts(texts, target_len=900, num_contexts=30):
    """Concatenate WikiText sentences into long ~900-token contexts."""
    gen = GPT2Generator("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    contexts = []
    buf = ""
    for t in texts:
        buf += " " + t
        tokens = gen.tokenizer.encode(buf, return_tensors="pt")
        if tokens.shape[1] >= target_len:
            truncated = gen.tokenizer.decode(tokens[0, :target_len])
            contexts.append(truncated)
            buf = ""
            if len(contexts) >= num_contexts:
                break
    del gen
    torch.cuda.empty_cache()
    return contexts


def run_long_context_benchmark(output_path="results/real_perplexity_long.json"):
    texts = load_wikitext(max_samples=200)
    contexts = build_long_contexts(texts, target_len=900, num_contexts=30)
    print(f"Built {len(contexts)} long contexts (~900 tokens each)")

    generator = GPT2Generator("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    tau_values = [0.0, 0.7, 0.8, 0.9, 0.95]
    results = []

    for tau in tau_values:
        ppl_uncomp, ppl_comp, ratios = [], [], []
        for text in tqdm(contexts, desc=f"tau={tau}"):
            try:
                r = generator.compute_perplexity(text, tau=tau)
                ppl_uncomp.append(r["ppl_uncompressed"])
                ppl_comp.append(r["ppl_compressed"])
                ratios.append(r["compression_ratio"])
            except Exception:
                continue

        if not ppl_uncomp:
            continue

        avg_uncomp = float(np.mean(ppl_uncomp))
        avg_comp = float(np.mean(ppl_comp))
        avg_ratio = float(np.mean(ratios))
        delta_pct = (avg_comp - avg_uncomp) / avg_uncomp * 100

        results.append({
            "tau": tau, "samples": len(ppl_uncomp),
            "ppl_uncompressed": avg_uncomp, "ppl_compressed": avg_comp,
            "ppl_delta": avg_comp - avg_uncomp,
            "ppl_delta_pct": delta_pct, "compression_ratio": avg_ratio,
        })
        print(f"tau={tau:.2f}: {len(ppl_uncomp)} ctx, uncomp={avg_uncomp:.1f}, "
              f"comp={avg_comp:.1f}, delta={delta_pct:+.2f}%, ratio={avg_ratio:.2f}x")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    return results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_long_context_benchmark()
