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

def calculate_epsilon(output_path, audit_path, ckpt_dir, delta=1e-5, dp_beta=1.0, m=1):
    """
    Calculate the total privacy budget (epsilon) using data-dependent accounting
    and optimize alpha for the tightest possible bound.
    """
    if not os.path.exists(output_path):
        print(f"Error: Output file not found at {output_path}")
        return

    with open(output_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    audit_data = None
    if os.path.exists(audit_path):
        with open(audit_path, 'r', encoding='utf-8') as f:
            audit_data = json.load(f)

    print(f"--- Optimized Privacy Budget Analysis ---")
    print(f"Loading Tokenizer from: {ckpt_dir} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    except Exception as e:
        tokenizer = None

    num_samples = len(output_data)
    
    # We will test a range of alpha values to find the minimum global epsilon
    alphas = [1.1, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0]
    best_epsilon = float('inf')
    best_theoretical_epsilon = float('inf')
    best_alpha = 0
    best_theoretical_alpha = 0
    
    using_realized = False
    all_sample_rdps = [] # Store per-token RDPs to re-calculate groups

    # 1. Collect all raw divergences
    all_raw_divs = []
    for i in range(num_samples):
        text = output_data[i]
        token_count = len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer else len(text)/4.0
        
        audit_entry = audit_data[i] if audit_data and i < len(audit_data) else None
        token_divs = audit_entry.get('b_module_token_divergences') if audit_entry else None
        
        if token_divs:
            using_realized = True
            all_raw_divs.append(token_divs)
        else:
            all_raw_divs.append([dp_beta] * int(token_count))

    # 2. Sweep Alpha to find the minimum Total Epsilon
    for alpha in alphas:
        total_eps_sum = 0
        total_theo_eps_sum = 0
        accountant_term = math.log(1.0 / delta) / (alpha - 1.0)
        
        for sample_divs in all_raw_divs:
            sample_rdp = 0
            sample_theo_rdp = 0
            for div in sample_divs:
                # Use current alpha in the formula
                sample_rdp += get_eps_step(div, alpha, m)
                sample_theo_rdp += get_eps_step(dp_beta, alpha, m)
                
            total_eps_sum += (sample_rdp + accountant_term)
            total_theo_eps_sum += (sample_theo_rdp + accountant_term)
        
        avg_eps = total_eps_sum / num_samples
        avg_theo_eps = total_theo_eps_sum / num_samples
        
        if avg_eps < best_epsilon:
            best_epsilon = avg_eps
            best_alpha = alpha
            
        if avg_theo_eps < best_theoretical_epsilon:
            best_theoretical_epsilon = avg_theo_eps
            best_theoretical_alpha = alpha

    print(f"Output File: {os.path.basename(output_path)}")
    print(f"Data Source: {'Audit File Found' if using_realized else 'No Audit File (Theoretical Only)'}")
    print(f"Number of Samples: {num_samples}")
    print(f"Target Delta: {delta}")
    
    print(f"\n[Theoretical Privacy Guarantee]")
    print(f"Optimal Rényi Order (Alpha): {best_theoretical_alpha}")
    print(f"Average Bound Epsilon (Theoretical Limit): {best_theoretical_epsilon:.2f}")

    if using_realized:
        print(f"\n[Realized Privacy Cost (Data-Dependent)]")
        print(f"Optimal Rényi Order (Alpha): {best_alpha}")
        print(f"Average Consumed Epsilon (Actual Utility): {best_epsilon:.2f}")

    print(f"-----------------------------------------\n")

    
    return best_epsilon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True, help="Path to outputs-*.json")
    parser.add_argument("--audit_file", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    
    args = parser.parse_args()
    calculate_epsilon(args.output_file, args.audit_file, args.ckpt_dir, args.delta, args.beta)
