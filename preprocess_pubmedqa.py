import json
import os
from datasets import load_dataset

def preprocess_pubmedqa():
    print("Downloading PubMedQA (pqa_labeled)...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    data_dir = "Data/pubmedqa"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Create pubmedqa.txt for indexing (the knowledge base)
    # We use the context and long_answer as the ground truth knowledge
    index_path = os.path.join(data_dir, "pubmedqa.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            # Context is a list of strings
            context = " ".join(item['context']['contexts'])
            entry = f"Context: {context}\nQuestion: {item['question']}\nAnswer: {item['long_answer']}\nLabel: {item['final_decision']}"
            f.write(entry + "\n\n")
            
    # 2. Create evaluation pairs (questions and ground truth answers)
    eval_input = [item['question'] for item in dataset]
    eval_output = [item['final_decision'] for item in dataset]
    
    with open(os.path.join(data_dir, "eval_input.json"), "w", encoding="utf-8") as f:
        json.dump(eval_input, f, indent=4)
        
    with open(os.path.join(data_dir, "eval_output.json"), "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=4)
        
    print(f"Preprocessed {len(dataset)} samples. Data saved in {data_dir}")

if __name__ == "__main__":
    preprocess_pubmedqa()
