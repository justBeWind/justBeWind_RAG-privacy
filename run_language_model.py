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
        {"role": "system", "content": """You are a privacy anonymization assistant.
Extract sensitive entities (names, meds, conditions, locations) and provide a NATURAL LANGUAGE generalization.
Example generalizations: 'a patient', 'a medical condition', 'a care facility', 'the medication', 'a measured value'.
IGNORE common English words and punctuation.

Format: JSON array [{"entity": "word", "type": "generalization"}]
If no entities found, output []."""},
        {"role": "user", "content": "Text: Hi, I am John. My GFR test at City Hospital was 45 mL/min."},
        {"role": "assistant", "content": '[{"entity": "John", "type": "a patient"}, {"entity": "GFR", "type": "a blood test"}, {"entity": "City Hospital", "type": "a care facility"}, {"entity": "45 mL/min", "type": "a measured value"}]'},
        {"role": "user", "content": f"Text: {text}"}
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"System: {messages[0]['content']}\n\nUser: {text}\n\nAssistant: ["

    if not prompt.strip().endswith("["):
        prompt += "["

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    resp_raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    resp = "[" + resp_raw if not resp_raw.strip().startswith("[") else resp_raw
    if "]" not in resp:
        last_brace = resp.rfind("}")
        if last_brace != -1:
            resp = resp[:last_brace+1] + "]"

    safe_text = text
    found_entities = []
    
    stop_words = {'i', 'me', 'my', 'am', 'is', 'a', 'an', 'the', 'hey', 'hi', 'it', 'doing', 'this', 'that', 'with', 'was', 'for', 'of', 'and', 'to', 'in', 'on', 'at'}
    for match in re.finditer(r'\{[^{}]*\}', resp):
        try:
            d_str = match.group().replace("'", '"')
            ent = json.loads(d_str)
            if isinstance(ent, dict) and 'entity' in ent and 'type' in ent:
                ent_text = str(ent['entity']).strip()
                if len(ent_text) <= 1: continue 
                if ent_text.lower() in stop_words: continue 
                found_entities.append(ent)
        except Exception:
            pass
            
    if found_entities:
        found_entities.sort(key=lambda x: len(str(x['entity'])), reverse=True)
        print(f"\n[AUDIT C-Module] Found {len(found_entities)} entities for semantic generalization.", flush=True)
        for ent in found_entities:
            entity_text = str(ent['entity']).strip()
            # Generalization is now a natural phrase
            general_phrase = str(ent['type']).strip() 
            pattern = r'(?<![a-zA-Z0-9])' + re.escape(entity_text) + r'(?![a-zA-Z0-9])'
            safe_text = re.sub(pattern, general_phrase, safe_text)
    else:
        if "[]" in resp or "{}" not in resp:
            print(f"\n[AUDIT C-Module] Success: No specific entities requiring redaction.")
        else:
            print(f"\n[AUDIT C-Module Warning] Heavy parsing failure. Fallback to Full Redaction.", flush=True)
            safe_text = "[REDACTED_CONTEXT_DUE_TO_PARSE_ERROR]"
            
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
        no_dp_rag: bool = False,
):
    print(f"\n[INIT] Single-Model Inference started for path: {path}")
    if no_dp_rag:
        print("Mode: BASELINE RAG (No Privacy Protection)")
    else:
        print(f"Mode: DP-RAG (Alpha: {dp_alpha}, Beta: {dp_beta})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Loading HuggingFace model: {ckpt_dir} on {device}...", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    inputs_dir = f"./Inputs&Outputs/{path}"
    with open(os.path.join(inputs_dir, "prompts.json"), 'r', encoding='utf-8') as f:
        all_prompts = json.loads(f.read())
        
    print(f"[INIT] Loaded {len(all_prompts)} prompts. Starting generation...", flush=True)
    
    answer_log = []
    audit_log = []
    ckpt_safe_name = ckpt_dir.split('/')[-1]
    
    suffix = "-baseline" if no_dp_rag else ""
    out_name = f"{inputs_dir}/outputs-{ckpt_safe_name}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}{suffix}.json"
    audit_name = f"{inputs_dir}/audit-{ckpt_safe_name}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}{suffix}.json"
    
    for i, prompt_priv in tqdm(enumerate(all_prompts), total=len(all_prompts), desc="Inference Pipeline"):
        
        start_time = time.time()
        
        if no_dp_rag:
            # Baseline: Just standard generation on the original prompt
            inputs = tokenizer(prompt_priv, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_gen_len, 
                    temperature=temperature, 
                    top_p=top_p, 
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id
                )
            ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            avg_lam = 1.0
            c_entities = []
            status = "BASELINE"
        else:
            # DP-RAG: C-Module + B-Module
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
                
            ans, avg_lam = dp_fusion_generate(model, tokenizer, prompt_priv, prompt_pub, device, max_gen_len, temperature, top_p, alpha=dp_alpha, max_div=dp_beta)
            status = "SUCCESS" if c_entities else "FALLBACK_REDACTED"

        end_time = time.time()
        gen_time = end_time - start_time
        
        print(f"\n[Output #{i+1}] Length: {len(ans)} chars. Time: {gen_time:.2f}s", flush=True)
        answer_log.append(ans)
        
        audit_log.append({
            "prompt_index": i + 1,
            "status": status,
            "c_module_entities_extracted": len(c_entities),
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
