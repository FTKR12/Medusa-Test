import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Medusa Test")
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--prompt_txt', default='')
    parser.add_argument('--prompt_path', default='')
    parser.add_argument('--model_name_or_path', default='huggingface/model/name/or/your/model/path', help='hf model name of model path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default='234', type=int)
    
    args = parser.parse_args()
    return args