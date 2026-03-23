import json
import math
import argparse
import os
from transformers import AutoTokenizer

def calculate_epsilon(output_path, ckpt_dir, alpha=2.0, delta=1e-5, dp_beta=1.0, m=1):
    """
    Calculate the total privacy budget (epsilon) using exact token counting.
    """
    if not os.path.exists(output_path):
        print(f"Error: Output file not found at {output_path}")
        return

    # Load outputs (containing actual generated text strings)
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("Error: Output file is empty.")
        return

    print(f"--- Precise Privacy Budget Analysis ---")
    print(f"Loading Tokenizer from: {ckpt_dir} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Falling back to 4 chars/token heuristic.")
        tokenizer = None

    # 1. Calculate Exact Response Length (T)
    total_tokens = 0
    num_samples = len(data)
    
    for i, text in enumerate(data):
        if tokenizer:
            # Count tokens for this specific output string
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
        else:
            token_count = len(text) / 4.0 # Heuristic fallback
        total_tokens += token_count
        
    avg_tokens = total_tokens / num_samples
    
    print(f"Output File: {os.path.basename(output_path)}")
    print(f"Number of Samples: {num_samples}")
    print(f"Avg Answer Length (T): {avg_tokens:.2f} tokens")
    
    # 2. Compute eps_step (Formula from Paper)
    eps_step = (1.0 / (alpha - 1.0)) * math.log(
        ((m - 1.0) / m) + (1.0 / m) * math.exp((alpha - 1.0) * 4.0 * dp_beta)
    )
    
    # 3. Total Epsilon
    accountant_term = math.log(1.0 / delta) / (alpha - 1.0)
    total_epsilon = avg_tokens * eps_step + accountant_term
    
    print(f"Parameters: Alpha={alpha}, Beta={dp_beta}, Delta={delta}, Groups={m}")
    print(f"Per-token Budget (eps_step): {eps_step:.4f}")
    print(f"Total Experiment Epsilon (Avg): {total_epsilon:.2f}")
    print(f"-----------------------------------------\n")
    
    return total_epsilon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True, help="Path to outputs-*.json")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to model checkpoint (for tokenizer)")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    
    args = parser.parse_args()
    calculate_epsilon(args.output_file, args.ckpt_dir, args.alpha, args.delta, args.beta)
