import json
import os
from datasets import load_dataset

def preprocess_pubmedqa(num_eval=250):
    print(f"Downloading PubMedQA (pqa_labeled)... Target evaluation samples: {num_eval}")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # 1. Create pubmedqa for indexing (the knowledge base)
    # We index ALL labeled contexts (1000) to provide a rich retrieval pool.
    index_dir = "Data/pubmedqa"
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "pubmedqa.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            context = " ".join(item['context']['contexts'])
            f.write(context + "\n\n")
            
    # 2. Create pubmedqa-test for evaluation pairs
    # We only take the first `num_eval` samples for inference to save time.
    test_dir = "Data/pubmedqa-test"
    os.makedirs(test_dir, exist_ok=True)
    
    eval_dataset = dataset.select(range(min(num_eval, len(dataset))))
    eval_input = [item['question'] for item in eval_dataset]
    eval_output = [item['final_decision'] for item in eval_dataset]
    
    with open(os.path.join(test_dir, "eval_input.json"), "w", encoding="utf-8") as f:
        json.dump(eval_input, f, indent=4)
        
    with open(os.path.join(test_dir, "eval_output.json"), "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=4)
        
    print(f"Preprocessed {len(dataset)} samples for indexing.")
    print(f"Created {len(eval_input)} samples for evaluation inference.")
    print(f"Indexing data saved in {index_dir}")
    print(f"Evaluation data saved in {test_dir}")

if __name__ == "__main__":
    # You can change this number to control how many questions to infer
    preprocess_pubmedqa(num_eval=250)
