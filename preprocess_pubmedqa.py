import json
import os
from datasets import load_dataset

def preprocess_pubmedqa():
    print("Downloading PubMedQA (pqa_labeled)...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    data_dir = "Data/pubmedqa-test"
    # 1. Create pubmedqa for indexing (the knowledge base)
    index_dir = "Data/pubmedqa"
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "pubmedqa.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            context = " ".join(item['context']['contexts'])
            f.write(context + "\n\n")
            
    # 2. Create pubmedqa-test for evaluation pairs
    test_dir = "Data/pubmedqa-test"
    os.makedirs(test_dir, exist_ok=True)
    eval_input = [item['question'] for item in dataset]
    eval_output = [item['final_decision'] for item in dataset]
    
    with open(os.path.join(test_dir, "eval_input.json"), "w", encoding="utf-8") as f:
        json.dump(eval_input, f, indent=4)
        
    with open(os.path.join(test_dir, "eval_output.json"), "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=4)
        
    print(f"Preprocessed {len(dataset)} samples.")
    print(f"Indexing data saved in {index_dir}")
    print(f"Evaluation data saved in {test_dir}")

if __name__ == "__main__":
    preprocess_pubmedqa()
