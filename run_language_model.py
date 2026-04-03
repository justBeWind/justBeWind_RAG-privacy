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

# ── C Module v2: Typed Placeholder + E_query Filtering ──────────────────────
# Allowed entity type labels (must match prompt exactly)
ALLOWED_TYPES = {
    "PERSON", "PHONE", "EMAIL", "ADDRESS", "AGE",
    "DISEASE", "MEDICATION", "TEST_RESULT", "DATE", "ID", "ORGANIZATION"
}

def get_typed_placeholder_context(context_text: str, query_text: str, model, tokenizer, device):
    """
    New C Module: NER with typed placeholders.
    Implements E_target = E_all - E_query:
      - Identifies all private entities in context_text
      - Filters out entities that already appear in query_text (user already knows them)
      - Replaces remaining entities with <TYPE> labels (e.g., <PERSON>, <DISEASE>)
    No semantic generalization is performed — raw type label only.
    This deliberately creates an information gap that gives DP-Fusion (B module) a meaningful role.
    """
    # Build few-shot prompt
    system_prompt = """You are a clinical privacy entity detector for RAG context sanitization.
Identify sensitive entities in the CONTEXT that should be protected.
Output ONLY their exact text and type label. Do NOT paraphrase or generalize.

ALLOWED TYPE LABELS (use EXACTLY one of these strings):
PERSON, PHONE, EMAIL, ADDRESS, AGE, DISEASE, MEDICATION, TEST_RESULT, DATE, ID, ORGANIZATION

RULES:
1. Only flag entities NOT already mentioned in the user's QUERY (the user already knows those).
2. DO NOT mask medical professional roles or titles (e.g., "neurosurgeon", "doctor", "specialist") unless they are specific proper names.
3. Specific lab values with numbers are TEST_RESULT. Generic medical concepts ("blood pressure") are NOT entities.
4. Identification must be EXHAUSTIVE: find every mention of the same entity to ensure full protection.
5. Output ONLY a valid JSON array: [{"entity": "exact text", "type": "TYPENAME"}, ...]
6. If nothing to protect, output []"""

    few_shot_query = "What should I do about my diabetes?"
    few_shot_context = "Patient John (age 52, 138-xxxx) was diagnosed with diabetes and visited his neurosurgeon at City Clinic last Jan 5. John mentioned his metformin 500mg dose."
    few_shot_output = '[{"entity": "John", "type": "PERSON"}, {"entity": "52", "type": "AGE"}, {"entity": "138-xxxx", "type": "PHONE"}, {"entity": "metformin 500mg", "type": "MEDICATION"}, {"entity": "Jan 5", "type": "DATE"}, {"entity": "City Clinic", "type": "ORGANIZATION"}]'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"QUERY: {few_shot_query}\nCONTEXT: {few_shot_context}"},
        {"role": "assistant", "content": few_shot_output},
        {"role": "user", "content": f"QUERY: {query_text}\nCONTEXT: {context_text}"}
    ]

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"System: {system_prompt}\n\nUser: QUERY: {query_text}\nCONTEXT: {context_text}\n\nAssistant: ["

    if not prompt.strip().endswith("["):
        prompt += "["

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    resp_raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    resp = "[" + resp_raw if not resp_raw.strip().startswith("[") else resp_raw
    if "]" not in resp:
        last_brace = resp.rfind("}")
        if last_brace != -1:
            resp = resp[:last_brace + 1] + "]"

    # ── Parse raw entity list from LLM output ────────────────────────────────
    stop_words = {'i', 'me', 'my', 'am', 'is', 'a', 'an', 'the', 'hey', 'hi', 'it',
                  'this', 'that', 'with', 'was', 'for', 'of', 'and', 'to', 'in', 'on', 'at'}
    all_entities = []
    for match in re.finditer(r'\{[^{}]*\}', resp):
        try:
            d_str = match.group().replace("'", '"')
            ent = json.loads(d_str)
            if isinstance(ent, dict) and 'entity' in ent and 'type' in ent:
                ent_text = str(ent['entity']).strip()
                ent_type = str(ent['type']).strip().upper()
                if len(ent_text) <= 1: continue
                if ent_text.lower() in stop_words: continue
                if ent_type not in ALLOWED_TYPES:
                    # Snap closest match or skip
                    continue
                all_entities.append({"entity": ent_text, "type": ent_type})
        except Exception:
            pass

    # ── E_query Filter: E_target = E_all \ E_query ───────────────────────────
    # Any entity whose text appears (case-insensitive) in the user's query is
    # already "public" – the user themselves disclosed it – so we do NOT mask it.
    query_lower = query_text.lower()
    target_entities = [
        ent for ent in all_entities
        if str(ent['entity']).lower() not in query_lower
    ]
    filtered_out = len(all_entities) - len(target_entities)

    # ── Replace with <TYPE> placeholders ─────────────────────────────────────
    typed_context = context_text
    if target_entities:
        # Sort by length descending to avoid partial-replacement bugs
        target_entities.sort(key=lambda x: len(str(x['entity'])), reverse=True)
        print(f"\n[AUDIT C-Module v2] {len(all_entities)} entities detected, "
              f"{filtered_out} filtered by E_query, "
              f"{len(target_entities)} replaced with type labels.", flush=True)
        for ent in target_entities:
            entity_text = str(ent['entity']).strip()
            placeholder = f"<{ent['type']}>"
            pattern = r'(?<![a-zA-Z0-9])' + re.escape(entity_text) + r'(?![a-zA-Z0-9])'
            typed_context = re.sub(pattern, placeholder, typed_context, flags=re.IGNORECASE)
    else:
        if not all_entities:
            print(f"\n[AUDIT C-Module v2] No sensitive entities found in context.", flush=True)
        else:
            print(f"\n[AUDIT C-Module v2] All {len(all_entities)} entities were in query (E_query). "
                  f"Context unchanged.", flush=True)

    return typed_context, target_entities

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
    divergences_log = []
    
    logits_priv = out_priv.logits[:, -1, :]
    logits_pub = out_pub.logits[:, -1, :]
    
    tau = max(0.1, temperature)
    
    for step in range(max_gen_len):
        p_priv = F.softmax(logits_priv / tau, dim=-1)
        p_pub = F.softmax(logits_pub / tau, dim=-1)
        
        lam, div = find_lambda_bisection(p_priv, p_pub, alpha, max_div)
        p_mixed = lam * p_priv + (1.0 - lam) * p_pub
        
        if temperature > 0:
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(p_mixed, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                p_mixed[indices_to_remove] = 0.0
                p_mixed = p_mixed / p_mixed.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(p_mixed, num_samples=1)
        else:
            next_token = torch.argmax(p_mixed, dim=-1).unsqueeze(0)
        
        generated_token = next_token.item()
        generated_tokens.append(generated_token)
        lambdas_log.append(lam)
        divergences_log.append(div)
        
        if generated_token == eos_id:
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
    return text, avg_lam, divergences_log

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
        baseline_only: bool = False,
        c_module_only: bool = False,
        max_test_samples: int = -1,
):
    print(f"\n[INIT] Single-Model Inference started for path: {path}")
    if baseline_only:
        print("Mode: BASELINE RAG (No Privacy Protection)")
    elif c_module_only:
        print("Mode: C-MODULE ONLY (Typed Placeholder, v2)")
    else:
        print(f"Mode: DP-RAG with Typed Placeholder C-Module (Alpha: {dp_alpha}, Beta: {dp_beta})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Loading HuggingFace model: {ckpt_dir} on {device}...", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    inputs_dir = f"./Inputs&Outputs/{path}"
    with open(os.path.join(inputs_dir, "prompts.json"), 'r', encoding='utf-8') as f:
        all_prompts = json.loads(f.read())
        
    if max_test_samples > 0:
        all_prompts = all_prompts[:max_test_samples]
        
    print(f"[INIT] Loaded {len(all_prompts)} prompts. Starting generation...", flush=True)
    
    answer_log = []
    audit_log = []
    ckpt_safe_name = ckpt_dir.split('/')[-1]
    
    if baseline_only:
        suffix = "-baseline"
    elif c_module_only:
        suffix = "-c_only"
    else:
        suffix = f"-dp_{dp_alpha}_{dp_beta}"

    out_name = f"{inputs_dir}/outputs-{ckpt_safe_name}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}{suffix}.json"
    audit_name = f"{inputs_dir}/audit-{ckpt_safe_name}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}{suffix}.json"
    
    for i, prompt_priv in tqdm(enumerate(all_prompts), total=len(all_prompts), desc="Inference Pipeline"):
        
        start_time = time.time()
        
        if baseline_only:
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
            prompt_pub = prompt_priv # In baseline, public prompt is the same as private
        else:
            # DP-RAG: C-Module (Typed Placeholder + E_query Filter) + B-Module
            start_marker = "context: "
            end_marker = "\nquestion: "
            start_idx = prompt_priv.find(start_marker)

            # Extract query text for E_query filtering (E_target = E_all \ E_query)
            question_marker = "question: "
            q_start = prompt_priv.find(question_marker)
            if q_start != -1:
                q_start += len(question_marker)
                # Question ends at the next blank line or answer prompt
                q_end = prompt_priv.find("\n", q_start)
                query_str = prompt_priv[q_start:q_end].strip() if q_end != -1 else prompt_priv[q_start:].strip()
            else:
                query_str = ""

            if start_idx != -1 and (end_idx := prompt_priv.find(end_marker, start_idx)) != -1:
                start_idx += len(start_marker)
                context_str = prompt_priv[start_idx:end_idx]
                typed_context, c_entities = get_typed_placeholder_context(
                    context_str, query_str, model, tokenizer, device
                )
                prompt_pub = prompt_priv[:start_idx] + typed_context + prompt_priv[end_idx:]
            else:
                # Fallback: treat entire prompt as context
                typed_context, c_entities = get_typed_placeholder_context(
                    prompt_priv, query_str, model, tokenizer, device
                )
                prompt_pub = typed_context

            if c_module_only:
                # C-Module Only: Standard generation on the typed-placeholder prompt
                inputs = tokenizer(prompt_pub, return_tensors="pt").to(device)
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
                divs = []
                status = "C_MODULE_ONLY_TYPED"
            else:
                # Full DP-RAG: B-Module (DP-Fusion) on D_priv vs D_pub (typed placeholder)
                ans, avg_lam, divs = dp_fusion_generate(
                    model, tokenizer, prompt_priv, prompt_pub, device,
                    max_gen_len, temperature, top_p, alpha=dp_alpha, max_div=dp_beta
                )

                if c_entities:
                    status = "SUCCESS_TYPED"
                else:
                    status = "NO_ENTITIES_FOUND"

        end_time = time.time()
        gen_time = end_time - start_time
        
        print(f"\n[Output #{i+1}] Length: {len(ans)} chars. Time: {gen_time:.2f}s", flush=True)
        answer_log.append(ans)
        
        audit_log.append({
            "prompt_index": i + 1,
            "status": status,
            "c_module_entities_extracted": len(c_entities),
            "c_module_entities_details": c_entities,
            "b_module_fusion_lambda_avg": round(avg_lam, 4),
            "b_module_token_divergences": [round(d, 6) for d in divs] if not baseline_only else [],
            "b_module_prompt_privacy": prompt_priv,
            "b_module_prompt_public": prompt_pub,
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
