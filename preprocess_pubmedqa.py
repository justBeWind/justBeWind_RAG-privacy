import json
import os
from datasets import load_dataset

def preprocess_pubmedqa(num_eval=250):
    print("Loading PubMedQA splits...")
    # Load labeled (1000) and unlabeled (61.2k)
    labeled_data = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    unlabeled_data = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
    
    index_dir = "Data/pubmedqa"
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "pubmedqa.txt")
    
    total_docs = 0
    print("Building purely clean Knowledge Base (Abstracts ONLY)...")
    
    # 1. 构建完全纯净的知识库 (写入所有的未标签数据 + 标签数据的摘要部分)
    with open(index_path, "w", encoding="utf-8") as f:
        # 写入 61.2k 未标签的医学摘要
        for item in unlabeled_data:
            context = " ".join(item['context']['contexts'])
            # 如果极端情况下混入了 Label，跳过保障安全
            if "Label:" in context or "Answer:" in context:
                continue
            f.write(context.replace('\n', ' ') + "\n\n")
            total_docs += 1
            
        # 写入 1000 条标签数据的医学摘要（这是 RAG 需要检索的内容）
        for item in labeled_data:
            context = " ".join(item['context']['contexts'])
            f.write(context.replace('\n', ' ') + "\n\n")
            total_docs += 1
            
    print(f"Success! Built Knowledge Base with {total_docs} pure medical abstracts.")
    
    # 2. 制作单纯用来 Eval 的 Question 和 Label
    test_dir = "Data/pubmedqa-test"
    os.makedirs(test_dir, exist_ok=True)
    
    eval_input = []
    eval_output = []
    
    # 我们依然只取前 num_eval (250) 个样本用来推理评测
    for i, item in enumerate(labeled_data):
        if i >= num_eval:
            break
        eval_input.append(item['question'])
        eval_output.append(item['final_decision'])
        
    with open(os.path.join(test_dir, "eval_input.json"), "w", encoding="utf-8") as f:
        json.dump(eval_input, f, indent=4)
        
    with open(os.path.join(test_dir, "eval_output.json"), "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=4)
        
    print(f"Saved {len(eval_input)} evaluation questions to {test_dir}.")

if __name__ == "__main__":
    preprocess_pubmedqa(num_eval=250)
