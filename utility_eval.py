import json
import argparse
import os
from rouge_score import rouge_scorer
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

def calculate_utility(ground_truth_path, predicted_path):
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    with open(predicted_path, 'r', encoding='utf-8') as f:
        predicted = json.load(f)
    
    if len(ground_truth) != len(predicted):
        print(f"Warning: Length mismatch! GT: {len(ground_truth)}, Pred: {len(predicted)}")
        min_len = min(len(ground_truth), len(predicted))
        ground_truth = ground_truth[:min_len]
        predicted = predicted[:min_len]

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    for gt, pred in zip(ground_truth, predicted):
        scores = scorer.score(gt, pred)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    print(f"Average ROUGE-L: {avg_rouge_l:.4f}")

    # BERTScore
    if BERT_SCORE_AVAILABLE:
        print("Calculating BERTScore...")
        P, R, F1 = bert_score(predicted, ground_truth, lang="en", verbose=True)
        avg_bert_f1 = F1.mean().item()
        print(f"Average BERTScore F1: {avg_bert_f1:.4f}")
    else:
        print("BERTScore not available. Please install with 'pip install bert-score'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, default="Data/chatdoctor-test/eval_output.json")
    parser.add_argument("--pred", type=str, required=True, help="Path to the RAG generated outputs-*.json")
    args = parser.parse_args()
    
    calculate_utility(args.gt, args.pred)
