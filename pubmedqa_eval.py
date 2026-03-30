import json
import argparse
import re

def parse_label(answer):
    answer = answer.lower().strip()
    
    # 1. 优先尝试匹配句子开头或开头的第一个有效词
    # 匹配像 "yes,", " yes\n", "**yes**" 等情况
    match = re.match(r'^[^a-z]*(yes|no|maybe)\b', answer)
    if match:
        return match.group(1)
        
    # 2. 如果开头没找到，退一步寻找整个回答中【第一个】出现的结果
    # 因为我们的 Prompt 要求模型 "Directly answer with YES, NO or MAYBE..."，所以它通常会放在前面。
    # 原代码中使用 matches[-1] 会错误地提取出解释文案里的 "no complications" 之类的词。
    matches = list(re.finditer(r'\b(yes|no|maybe)\b', answer))
    if matches:
        return matches[0].group(1) 
        
    return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted JSON outputs")
    parser.add_argument("--ans", type=str, default="Data/pubmedqa-test/eval_output.json", help="Path to ground truth JSON")
    args = parser.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        preds = json.load(f)
    with open(args.ans, "r", encoding="utf-8") as f:
        gts = json.load(f)

    num_samples = min(len(preds), len(gts))
    correct = 0
    results = []

    for p, g in zip(preds[:num_samples], gts[:num_samples]):
        pred_label = parse_label(p)
        if pred_label == g.lower():
            correct += 1
        results.append({
            "pred": p,
            "pred_label": pred_label,
            "gt_label": g
        })

    accuracy = correct / num_samples
    print(f"Accuracy: {accuracy:.4f} ({correct}/{num_samples})")

    # Save details
    out_file = args.pred.replace(".json", "_accuracy.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "details": results}, f, indent=4)
    print(f"Detailed accuracy results saved to {out_file}")

if __name__ == "__main__":
    main()
