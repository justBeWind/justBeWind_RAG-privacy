import json
import os
from nltk.tokenize import RegexpTokenizer

try:
    tokenizer = RegexpTokenizer(r'\w+')
    # 请确认这个路径在服务器上是存在的，或者修改为正确的相对路径
    path = 'Inputs&Outputs/chat-target/Q-R-T-/outputs-Llama-2-7b-chat-hf-0.6-0.9-4096-256.json'

    if not os.path.exists(path):
        print(f"Error: {path} not found.")
    else:
        with open(path, 'r', encoding='utf-8') as f:
            outputs = json.load(f)

        print(f"Total outputs loaded: {len(outputs)}")

        for i in range(min(10, len(outputs))):
            text = outputs[i]
            toks = tokenizer.tokenize(text)
            print(f"\n--- Index {i} ---")
            print(f"Character Length: {len(text)}")
            print(f"Token Count: {len(toks)}")
            print(f"Text Content (First 100 chars): {repr(text[:100])}")
            print(f"First 15 Tokens: {toks[:15]}")
except Exception as e:
    print(f"An error occurred: {e}")
