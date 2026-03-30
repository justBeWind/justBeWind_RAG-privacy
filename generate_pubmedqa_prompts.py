import json
import os
from generate_prompt import get_prompt, get_executable_file
from retrieval_database import construct_retrieval_database

def main():
    exp_name = 'pubmedqa-v2'
    os.makedirs(f'Inputs&Outputs/{exp_name}', exist_ok=True)
    
    # 1. Construct Retrieval Database for PubMedQA safely
    db_path = './RetrievalBase/pubmedqa/bge-large-en-v1.5'
    if not os.path.exists(db_path) or len(os.listdir(db_path)) == 0:
        print("Vector Database for PubMedQA not found. Building now...")
        construct_retrieval_database(['pubmedqa'], ['recursive_character'], 'bge-large-en-v1.5')
    else:
        print("Vector Database for PubMedQA already exists. Skipping Chroma construction to prevent duplication...")    
    # 2. Define Settings for PubMedQA Evaluation
    # We use Performance_pubmedqa to trigger the evaluation mode in generate_prompt.py
    settings = {
        'question': {
            'question_prefix': [''],
            'question_suffix': [''],
            'question_adhesive': [''],
            'question_infor': ['Performance_pubmedqa'] 
        },
        'retrival': {
            'data_name_list': [['pubmedqa']],
            'encoder_model_name': ['bge-large-en-v1.5'],
            'retrieve_method': ['knn'],
            'retrieve_num': [3],
            'contexts_adhesive': ['\n\n'],
            'threshold': [-1],
            'rerank': ['yes'],
            'summarize': ['no'],
            'num_questions': 1000, # PubMedQA labeled has 1000 samples
            'max_context_length': 2048
        },
        'template': {
            'suffix': [['context: ', 'question: ', 'Directly answer with YES, NO or MAYBE, and provide a short explanation.\nanswer:']],
            'template_adhesive': ['\n']
        },
        'LLM': {
            'LLM model': ['meta-llama/Llama-2-7b-chat-hf'],
            'temperature': [0.1],
            'top_p': [0.9],
            'max_seq_len': [4096],
            'max_gen_len': [128] # Binary-ish answers are short
        }
    }
    
    setting_path = f'Inputs&Outputs/{exp_name}/setting.json'
    with open(setting_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)

    GPU_available = '0' 
    master_port = 27000
    
    print(f'Generating prompts for {exp_name}...')
    output_list = get_prompt(settings, exp_name)
    get_executable_file(settings, exp_name, output_list, GPU_available, master_port)
    print(f'Successfully generated prompts and bash script for {exp_name}')

if __name__ == '__main__':
    # Ensure standard directory exists
    if not os.path.exists('Data/pubmedqa-test'):
        print("Error: Data/pubmedqa-test not found. Please run preprocess_pubmedqa.py first.")
    else:
        main()
