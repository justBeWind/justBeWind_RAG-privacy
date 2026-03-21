import fire
import json
import os
import torch
import torch.nn.functional as F
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def compute_renyi_divergence(p, q, alpha=2.0, eps=1e-10):
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    term_pq = torch.sum(p.pow(alpha) * q.pow(1.0 - alpha), dim=-1).clamp_min(eps)
    div_pq = (1.0 / (alpha - 1.0)) * torch.log(term_pq)
    
    term_qp = torch.sum(q.pow(alpha) * p.pow(1.0 - alpha), dim=-1).clamp_min(eps)
    div_qp = (1.0 / (alpha - 1.0)) * torch.log(term_qp)
    
    return torch.maximum(div_pq, div_qp)

def find_lambda_bisection(p_priv, p_pub, alpha, max_div, max_iter=20, tol=1e-5):
    if max_div <= 0: return 0.0, 0.0
    div_at_1 = compute_renyi_divergence(p_priv, p_pub, alpha)
    if div_at_1 <= max_div: return 1.0, div_at_1.item()
    
    left, right = 0.0, 1.0
    for _ in range(max_iter):
        mid = (left + right) / 2.0
        mix = mid * p_priv + (1.0 - mid) * p_pub
        div = compute_renyi_divergence(mix, p_pub, alpha)
        if div > max_div:
            right = mid
        else:
            left = mid
        if (right - left) < tol:
            break
    final_lambda = left
    final_mix = final_lambda * p_priv + (1.0 - final_lambda) * p_pub
    final_div = compute_renyi_divergence(final_mix, p_pub, alpha)
    return final_lambda, final_div.item()

def get_safe_context(text, model, tokenizer, device):
    messages = [
        {"role": "system", "content": "You are a privacy anonymization engine. Extract sensitive entities (names, hospitals, specific diseases, exact amounts, etc.) from the text. Generalize them to Intermediate-level Semantic Categories.\nFor example: 'John' -> '[PERSON]', 'Type-2 Diabetes' -> '[CHRONIC_METABOLIC_DISEASE]'. Do NOT use overly broad tags like [DISEASE] or [NUMBER].\nOutput MUST be a valid JSON array of objects: [{\"entity\": \"exact_word\", \"type\": \"[TAG]\"}]. NO other text."},
        {"role": "user", "content": f"Text: {text}"}
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for older models without explicit chat template
        prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:\n"
    
    prompt += "["  # Force the LLM to output a JSON array
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Prepend the missing bracket because we forced it in the prompt
    resp = "[" + resp
    
    safe_text = text
    try:
        json_str = re.search(r'\[.*\]', resp, re.DOTALL).group()
        entities = json.loads(json_str)
        print(f"[AUDIT C-Module] Found entities to generalize: {entities}", flush=True)
        for ent in entities:
            if isinstance(ent, dict) and 'entity' in ent and 'type' in ent:
                safe_text = safe_text.replace(str(ent['entity']), str(ent['type']))
    except Exception as e:
        print(f"[AUDIT C-Module Warning] JSON parsing failed: {resp}")
        print("[AUDIT C-Module Warning] Fallback: Heavily redacting context to ensure Absolute DP Guarantee.", flush=True)
        safe_text = "[REDACTED_CONTEXT_DUE_TO_PARSE_ERROR]"
    return safe_text

def dp_fusion_generate(model, tokenizer, prompt_priv, prompt_pub, device, max_gen_len, temperature, top_p, alpha=2.0, max_div=1.0):
    eos_id = tokenizer.eos_token_id
    
    input_priv = tokenizer(prompt_priv, return_tensors="pt").input_ids.to(device)
    input_pub = tokenizer(prompt_pub, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        out_priv = model(input_ids=input_priv, use_cache=True)
        out_pub = model(input_ids=input_pub, use_cache=True)
        
    past_priv = out_priv.past_key_values
    past_pub = out_pub.past_key_values
    
    generated_tokens = []
    lambdas_log = []
    
    logits_priv = out_priv.logits[:, -1, :]
    logits_pub = out_pub.logits[:, -1, :]
    
    tau = max(0.1, temperature)
    
    for step in range(max_gen_len):
        p_priv = F.softmax(logits_priv / tau, dim=-1)
        p_pub = F.softmax(logits_pub / tau, dim=-1)
        
        lam, div = find_lambda_bisection(p_priv, p_pub, alpha, max_div)
        p_mixed = lam * p_priv + (1.0 - lam) * p_pub
        
        # Greedy search for deterministic stability in attacks
        next_token = torch.argmax(p_mixed, dim=-1).unsqueeze(0)
        
        generated_tokens.append(next_token.item())
        lambdas_log.append(lam)
        
        if next_token.item() == eos_id:
            break
            
        with torch.no_grad():
            out_priv = model(input_ids=next_token, past_key_values=past_priv, use_cache=True)
            out_pub = model(input_ids=next_token, past_key_values=past_pub, use_cache=True)
            
        past_priv = out_priv.past_key_values
        past_pub = out_pub.past_key_values
        logits_priv = out_priv.logits[:, -1, :]
        logits_pub = out_pub.logits[:, -1, :]

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    avg_lam = sum(lambdas_log)/len(lambdas_log) if lambdas_log else 0
    return text, avg_lam

def main(
        ckpt_dir: str,
        path: str,
        tokenizer_path: str = '',
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 4096,
        max_gen_len: int = 256,
        max_batch_size: int = 1,
        dp_alpha: float = 2.0,
        dp_beta: float = 1.0,
):
    print(f"\n[INIT] Single-Model DP-RAG Inference started for path: {path}")
    print(f"Hyperparams -> Temp: {temperature}, DP_Alpha: {dp_alpha}, DP_Beta(Divergence_Bound): {dp_beta}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Loading HuggingFace model: {ckpt_dir} on {device}...", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    inputs_dir = f"./Inputs&Outputs/{path}"
    with open(os.path.join(inputs_dir, "prompts.json"), 'r', encoding='utf-8') as f:
        all_prompts = json.loads(f.read())
        
    print(f"[INIT] Loaded {len(all_prompts)} prompts. Starting generation with Auditing...", flush=True)
    
    try:
        with open(os.path.join(inputs_dir, "set.json"), "r", encoding='utf-8') as file:
            settings = json.load(file)
        suf = settings['suffix']
        adhesive = settings['adhesive_prompt']
        prefix_len = len(suf[0])
        end_marker = adhesive + suf[1]
    except Exception as e:
        print(f"[INIT Warning] Could not parse set.json fully. Assuming context falls back to entire prompt. {e}")
        suf, adhesive, prefix_len, end_marker = None, None, 0, None
    
    answer_log = []
    
    for i, prompt_priv in enumerate(all_prompts):
        print(f"\n=== Processing Prompt {i+1}/{len(all_prompts)} ===", flush=True)
        
        if end_marker and (end_idx := prompt_priv.find(end_marker, prefix_len)) != -1:
            context_str = prompt_priv[prefix_len:end_idx]
            print(f"[C-Module] Isolated RAG context. Length: {len(context_str)} characters.", flush=True)
            safe_context = get_safe_context(context_str, model, tokenizer, device)
            prompt_pub = prompt_priv.replace(context_str, safe_context)
        else:
            print(f"[C-Module] Isolated context not securely found. Running C-module on entire prompt.", flush=True)
            prompt_pub = get_safe_context(prompt_priv, model, tokenizer, device)
            
        start_time = time.time()
        ans, avg_lam = dp_fusion_generate(model, tokenizer, prompt_priv, prompt_pub, device, max_gen_len, temperature, top_p, alpha=dp_alpha, max_div=dp_beta)
        end_time = time.time()
        
        print(f"[B-Module Output] Length: {len(ans)} chars. Avg Fusion Lambda: {avg_lam:.4f}. Generation Time: {end_time - start_time:.2f}s", flush=True)
        answer_log.append(ans)
        
    ckpt_safe_name = ckpt_dir.split('/')[-1] # Ensure safe save path
    out_name = f"./Inputs&Outputs/{path}/outputs-{ckpt_safe_name}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json"
    
    with open(out_name, 'w', encoding='utf-8') as file:
        file.write(json.dumps(answer_log))
        
    print(f"\n[DONE] Successfully saved results to {out_name}", flush=True)

if __name__ == "__main__":
    fire.Fire(main)
