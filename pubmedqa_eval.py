import json
import argparse
import re

def parse_label(answer):
    # Search for yes/no/maybe in the answer
    answer = answer.lower()
    if re.search(r'\byes\b', answer):
        return "yes"
    if re.search(r'\bno\b', answer):
        return "no"
    if re.search(r'\bmaybe\b', answer):
        return "maybe"
    # Fallback if multiple are present or none are clear
    return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted JSON outputs")
    parser.add_argument("--gt", type=str, default="Data/pubmedqa/eval_output.json")
    args = parser.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        preds = json.load(f)
    with open(args.gt, "r", encoding="utf-8") as f:
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
