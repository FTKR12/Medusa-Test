import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Medusa Test")
    parser.add_argument('--name', default='medusa')
    parser.add_argument('--use_medusa', default=True, type=bool)
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--prompt_txt', default='')
    parser.add_argument('--prompt_path', default='')
    parser.add_argument('--model_path', default='huggingface/model/name/or/your/model/path', help='hf model name of model path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--num_out_tokens', default=128, type=int)
    
    args = parser.parse_args()
    return args