import os
import json
import argparse
import pandas as pd
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
except ImportError:
    print("Please install ragas: pip install ragas")
    exit(1)

try:
    from langchain_openai.chat_models import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    print("Please install langchain-openai: pip install langchain-openai")
    exit(1)


class CleanChatOpenAI(ChatOpenAI):
    """Custom wrapper to strip markdown JSON blocks from LLM output."""
    def _clean_content(self, content):
        if not isinstance(content, str):
            return content
        content = content.strip()
        # Very aggressive: find first { and last }
        import re
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            return match.group(1)
        return content

    def generate_prompt(self, *args, **kwargs):
        result = super().generate_prompt(*args, **kwargs)
        for chat_generations in result.generations:
            for gen in chat_generations:
                if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                    gen.message.content = self._clean_content(gen.message.content)
                elif hasattr(gen, 'text'):
                    gen.text = self._clean_content(gen.text)
        return result

    async def agenerate_prompt(self, *args, **kwargs):
        result = await super().agenerate_prompt(*args, **kwargs)
        for chat_generations in result.generations:
            for gen in chat_generations:
                if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                    gen.message.content = self._clean_content(gen.message.content)
                elif hasattr(gen, 'text'):
                    gen.text = self._clean_content(gen.text)
        return result


def load_data(pred_file, num_samples=30):
    base_dir = "Inputs&Outputs/chatdoctor-utility/Q-R-T-"
    # Load questions
    with open(f"{base_dir}/question.json", 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Load contexts
    with open(f"{base_dir}/context.json", 'r', encoding='utf-8') as f:
        contexts_raw = json.load(f)
        
    # Load predictions
    with open(pred_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)

    # Note: Ragas expects contexts to be List[str], our context.json 
    # has List[List[str]] so it matches exactly.
    
    # Optional: ground truth (not needed for faithfulness/answer_relevancy, but good to have)
    with open("Data/chatdoctor-test/eval_output.json", 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
        
    min_len = min(len(questions), len(contexts_raw), len(answers), len(ground_truths))
    
    # We slice to num_samples to save API cost during testing
    min_len = min(min_len, num_samples)
    print(f"Loaded {min_len} samples for evaluation from {pred_file}")

    dataset_dict = {
        "question": questions[:min_len],
        "contexts": contexts_raw[:min_len],
        "answer": answers[:min_len],
        "ground_truth": ground_truths[:min_len] # optional for these metrics
    }
    return Dataset.from_dict(dataset_dict)


def run_evaluation(dataset, output_csv="ragas_results.csv"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-ormpzusfphkhkvbwbgpfyskyejxqnbhtowgsaoidceeroiox")
    os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    
    print("Initializing Models via SiliconFlow API...")
    
    # Qwen 2.5 72B as Judge (most stable for JSON)
    judge_llm = CleanChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.0
    )
    
    # BGE embeddings via SiliconFlow (M3 supports up to 8192 tokens)
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        skip_empty=True
    )
    
    print("Starting RAGAS Evaluation...")
    # Setup metrics
    # Note: answer_relevancy often uses n > 1 which DeepSeek-V3 might not support.
    # We explicitly set n_generations to 1 to avoid this.
    try:
        # In newer ragas versions, metrics are classes.
        faithfulness.llm = judge_llm
        
        answer_relevancy.llm = judge_llm
        answer_relevancy.embeddings = embeddings
        answer_relevancy.n_generations = 1 
        
        metrics = [faithfulness, answer_relevancy]
        
        # Use a small batch_size to stay under SiliconFlow TPM limits
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=embeddings,
            raise_exceptions=True,
            batch_size=2 # Concurrency limit
        )
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=== Evaluation Complete ===")
    print(result)
    
    df = result.to_pandas()
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to {output_csv}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted JSON (outputs-*.json)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate (default 10 to save API costs)")
    parser.add_argument("--out", type=str, default="ragas_results.csv", help="Output CSV path")
    args = parser.parse_args()

    ds = load_data(args.pred, args.samples)
    run_evaluation(ds, args.out)

