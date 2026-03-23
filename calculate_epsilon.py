import json
import math
import argparse
import os
from transformers import AutoTokenizer

def get_eps_step(beta, alpha, m):
    """
    Calculate the per-token RDP cost based on DP-Fusion formula.
    """
    # term = (m-1)/m + (1/m) * exp((alpha-1) * 4 * beta)
    arg = ((m - 1.0) / m) + (1.0 / m) * math.exp((alpha - 1.0) * 4.0 * beta)
    return (1.0 / (alpha - 1.0)) * math.log(arg)

def calculate_epsilon(output_path, audit_path, ckpt_dir, alpha=2.0, delta=1e-5, dp_beta=1.0, m=1):
    """
    Calculate the total privacy budget (epsilon) using realized divergence if available,
    otherwise fallback to length-based theoretical bound.
    """
    if not os.path.exists(output_path):
        print(f"Error: Output file not found at {output_path}")
        return

    # Load outputs (containing actual generated text strings)
    with open(output_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    # Load audit data (containing per-token realized divergences)
    audit_data = None
    if os.path.exists(audit_path):
        with open(audit_path, 'r', encoding='utf-8') as f:
            audit_data = json.load(f)

    print(f"--- Precise Privacy Budget Analysis ---")
    print(f"Loading Tokenizer from: {ckpt_dir} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Falling back to 4 chars/token heuristic.")
        tokenizer = None

    num_samples = len(output_data)
    total_epsilon_sum = 0
    total_tokens_sum = 0
    
    accountant_term = math.log(1.0 / delta) / (alpha - 1.0)
    using_realized = False

    for i in range(num_samples):
        text = output_data[i]
        
        # 1. Get Token Count T
        if tokenizer:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
        else:
            token_count = len(text) / 4.0
        total_tokens_sum += token_count
        
        # 2. Calculate Epsilon for this sample
        sample_rdp_sum = 0
        
        # Check if audit has per-token divergences
        audit_entry = audit_data[i] if audit_data and i < len(audit_data) else None
        token_divs = audit_entry.get('b_module_token_divergences') if audit_entry else None
        
        if token_divs:
            using_realized = True
            for div in token_divs:
                sample_rdp_sum += get_eps_step(div, alpha, m)
        else:
            # Fallback to T * eps_step(dp_beta)
            sample_rdp_sum = token_count * get_eps_step(dp_beta, alpha, m)
            
        total_epsilon_sum += (sample_rdp_sum + accountant_term)

    avg_tokens = total_tokens_sum / num_samples
    avg_epsilon = total_epsilon_sum / num_samples
    
    print(f"Output File: {os.path.basename(output_path)}")
    print(f"Audit File: {os.path.basename(audit_path) if audit_path else 'None'}")
    print(f"Accounting Mode: {'DATA-DEPENDENT (REALIZED)' if using_realized else 'THEORETICAL (LENGTH-BASED)'}")
    print(f"Number of Samples: {num_samples}")
    print(f"Avg Answer Length (T): {avg_tokens:.2f} tokens")
    
    print(f"Parameters: Alpha={alpha}, Beta_Bound={dp_beta}, Delta={delta}, Groups={m}")
    print(f"Total Experiment Epsilon (Avg): {avg_epsilon:.2f}")
    print(f"-----------------------------------------\n")
    
    return avg_epsilon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True, help="Path to outputs-*.json")
    parser.add_argument("--audit_file", type=str, default="", help="Path to audit-*.json (for realized divergence)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    
    args = parser.parse_args()
    calculate_epsilon(args.output_file, args.audit_file, args.ckpt_dir, args.alpha, args.delta, args.beta)
