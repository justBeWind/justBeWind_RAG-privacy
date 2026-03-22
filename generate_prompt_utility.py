import json
import os
from generate_prompt import get_prompt, get_executable_file

def main():
    exp_name = 'chatdoctor-utility'
    setting_path = f'Inputs&Outputs/{exp_name}/setting.json'
    
    if not os.path.exists(setting_path):
        print(f"Error: {setting_path} not found!")
        return

    with open(setting_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    GPU_available = '0' 
    master_port = 27000
    
    print(f'Processing {exp_name} using {setting_path}')
    output_list = get_prompt(settings, exp_name)
    get_executable_file(settings, exp_name, output_list, GPU_available, master_port)
    print(f'Successfully generated prompts and bash script for {exp_name}')

if __name__ == '__main__':
    main()
