import fire
import json
import os
import torch
import torch.nn.functional as F
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm

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
        {"role": "system", "content": """You are defensive privacy anonymization engine. 
Identify and generalize sensitive entities while preserving utility.

CRITICAL RULES:
1. ONLY extract specific proper names, technical clinical terms, measurements, or unique identifiers.
2. DO NOT extract common English words (pronouns like 'I', 'me', common verbs like 'am', 'is', 'doing', or greetings like 'hi', 'hey').
3. Keep multi-word entities together (e.g., 'blood pressure', not 'blood' and 'pressure').
4. Map entities to general categories: [PERSON], [MEDICAL_TEST], [QUANTITATIVE_VALUE], [CONDITION], [MEDICATION], [LOCATION].

Output MUST be a valid JSON array: [{"entity": "exact_text", "type": "[TAG]"}]. 
Provide ONLY the JSON."""},
        {"role": "user", "content": "Text: Hi, I'm John. My GFR test at City Hospital was 45 mL/min today."},
        {"role": "assistant", "content": '[{"entity": "John", "type": "[PERSON]"}, {"entity": "GFR", "type": "[MEDICAL_TEST]"}, {"entity": "City Hospital", "type": "[LOCATION]"}, {"entity": "45 mL/min", "type": "[QUANTITATIVE_VALUE]"}]'},
        {"role": "user", "content": f"Text: {text}"}
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for models without chat template
        sys_p = messages[0]['content']
        user_p = messages[3]['content']
        prompt = f"System: {sys_p}\n\nUser: {user_p}\n\nAssistant: ["

    # If prompt doesn't end with "[", force it to encourage JSON array start
    if not prompt.strip().endswith("["):
        prompt += "["

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    resp_raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Re-normalize response to ensure it's a valid JSON array string
    resp = "[" + resp_raw if not resp_raw.strip().startswith("[") else resp_raw
    if not resp.strip().endswith("]"):
        # Look for the last } and add ]
        last_brace = resp.rfind("}")
        if last_brace != -1:
            resp = resp[:last_brace+1] + "]"

    safe_text = text
    found_entities = []
    
    # More robust JSON regex
    for match in re.finditer(r'\{[^{}]*\}', resp):
        try:
            d_str = match.group().replace("'", '"')
            ent = json.loads(d_str)
            if isinstance(ent, dict) and 'entity' in ent and 'type' in ent:
                # Basic Stop-word Filter for common 1-2 letter words or pronouns
                stop_words = {'i', 'me', 'my', 'am', 'is', 'a', 'an', 'the', 'hey', 'hi', 'it', 'doing', 'this', 'that'}
                if str(ent['entity']).lower().strip() not in stop_words:
                    found_entities.append(ent)
        except Exception:
            pass
            
    if found_entities:
        # Sort by length descending to avoid partial replacement (e.g., 'blood' before 'blood pressure')
        found_entities.sort(key=lambda x: len(str(x['entity'])), reverse=True)
        print(f"\n[AUDIT C-Module] Generalizing {len(found_entities)} entities...", flush=True)
        for ent in found_entities:
            entity_text = str(ent['entity'])
            tag = str(ent['type'])
            # Word-boundary aware replacement to avoid breaking words (like 'if' inside 'gift')
            pattern = r'\b' + re.escape(entity_text) + r'\b'
            safe_text = re.sub(pattern, tag, safe_text)
    else:
        # Only fallback to full redaction if the response was truly empty or unparseable
        if "[REDACTED]" not in resp and "{" not in resp:
            print(f"\n[AUDIT C-Module Warning] Heavy parsing failure. Fallback to Full Redaction.", flush=True)
            safe_text = "[REDACTED_CONTEXT_DUE_TO_PARSE_ERROR]"
        else:
            print(f"\n[AUDIT C-Module Warning] Partial parse success or empty entities.")
            
    return safe_text, found_entities

def dp_fusion_generate(model, tokenizer, prompt_priv, prompt_pub, device, max_gen_len, temperature, top_p, alpha=2.0, max_div=1.0):
    eos_id = tokenizer.eos_token_id
    
    input_priv = tokenizer(prompt_priv, return_tensors="pt").input_ids.to(device)
    input_pub = tokenizer(prompt_pub, return_tensors="pt").input_ids.to(device)
    
    attention_mask_priv = torch.ones_like(input_priv)
    attention_mask_pub = torch.ones_like(input_pub)
    
    with torch.no_grad():
        out_priv = model(input_ids=input_priv, attention_mask=attention_mask_priv, use_cache=True)
        out_pub = model(input_ids=input_pub, attention_mask=attention_mask_pub, use_cache=True)
        
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
        
        next_token = torch.argmax(p_mixed, dim=-1).unsqueeze(0)
        
        generated_tokens.append(next_token.item())
        lambdas_log.append(lam)
        
        if next_token.item() == eos_id:
            break
            
        attention_mask_priv = torch.cat([attention_mask_priv, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
        attention_mask_pub = torch.cat([attention_mask_pub, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
            
        with torch.no_grad():
            out_priv = model(input_ids=next_token, past_key_values=past_priv, attention_mask=attention_mask_priv, use_cache=True)
            out_pub = model(input_ids=next_token, past_key_values=past_pub, attention_mask=attention_mask_pub, use_cache=True)
            
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
    
    answer_log = []
    audit_log = []
    ckpt_safe_name = ckpt_dir.split('/')[-1]
    out_name = f"{inputs_dir}/outputs-{ckpt_safe_name}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json"
    audit_name = f"{inputs_dir}/audit-{ckpt_safe_name}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json"
    
    for i, prompt_priv in tqdm(enumerate(all_prompts), total=len(all_prompts), desc="DP-RAG Pipeline"):
        
        # Hardcoded exact string split based on A's RAG-privacy prompt format
        start_marker = "context: "
        end_marker = "\nquestion: "
        start_idx = prompt_priv.find(start_marker)
        
        if start_idx != -1 and (end_idx := prompt_priv.find(end_marker, start_idx)) != -1:
            start_idx += len(start_marker)
            context_str = prompt_priv[start_idx:end_idx]
            safe_context, c_entities = get_safe_context(context_str, model, tokenizer, device)
            prompt_pub = prompt_priv[:start_idx] + safe_context + prompt_priv[end_idx:]
        else:
            prompt_pub, c_entities = get_safe_context(prompt_priv, model, tokenizer, device)
            
        start_time = time.time()
        ans, avg_lam = dp_fusion_generate(model, tokenizer, prompt_priv, prompt_pub, device, max_gen_len, temperature, top_p, alpha=dp_alpha, max_div=dp_beta)
        end_time = time.time()
        
        gen_time = end_time - start_time
        print(f"\n[B-Module Output #{i+1}] Length: {len(ans)} chars. Fusion Weight Lambda: {avg_lam:.4f}. Generation Time: {gen_time:.2f}s", flush=True)
        answer_log.append(ans)
        
        # Incrementally log the Global Audit state
        # Enhancing snapshot with specific module status flags as per rigorous ablation requirements
        audit_log.append({
            "prompt_index": i + 1,
            "c_module_status": "SUCCESS" if c_entities else "FALLBACK_REDACTED",
            "c_module_entities_extracted": len(c_entities) if c_entities else 0,
            "c_module_entities_details": c_entities,
            "b_module_fusion_lambda_avg": round(avg_lam, 4),
            "b_module_output_length": len(ans),
            "generation_time_sec": round(gen_time, 2)
        })
        
        with open(out_name, 'w', encoding='utf-8') as file:
            file.write(json.dumps(answer_log, ensure_ascii=False, indent=2))
            
        with open(audit_name, 'w', encoding='utf-8') as a_file:
            a_file.write(json.dumps(audit_log, ensure_ascii=False, indent=2))
        
    print(f"\n[DONE] Successfully saved all results to {out_name}", flush=True)

if __name__ == "__main__":
    fire.Fire(main)
