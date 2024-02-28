import json
import time
import logging
import torch
from tqdm import tqdm
from typing import TypedDict, Tuple
from abc import abstractmethod
import logging

from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
from medusa.model.medusa_model import MedusaModel

# output structure of runner.run()
class LLMOutput(TypedDict):
    model_name: str
    input_sentence: str
    len_input_tokens: int
    inference_time: float
    output_sentence: str
    len_output_tokens: int
    throughput: float

class BaseRunner():
    def __init__(self, args):
        self.args = args
        
        # set prompt
        if len(args.prompt_path) != 0:
            with open(args.prompt_path, 'r') as f:
                self.prompt = f.read()
        else:
            self.prompt = args.prompt_txt
        assert len(self.prompt) !=0, "Set your prompt!"

        # set logger
        self.logger = logging.getLogger('Medusa Test')

        # TODO: write following variable
        self.llm = None
        self.tokenizer = None

    @abstractmethod
    def set_llm(self):
        pass


    @abstractmethod
    def run(self) -> LLMOutput:
        pass

    @abstractmethod
    def inference(self) -> Tuple[list, list, list, int, list, float]:
        pass

class DefaultLLMRunner():
    def __init__(self):
        pass

    def run(self):
        pass

class MedusaRunner(BaseRunner):
    def set_llm(self):
        self.llm = MedusaModel.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.tokenizer = self.llm.get_tokenizer()

    def run(self):
        self.set_llm()
        input_ids, input_len, output_ids, output_len, output_sentence, inference_time = self.inference()
        return LLMOutput(
            model_name=self.args.model_path,
            len_input_tokens=input_len,
            inference_time=inference_time,
            len_output_tokens=output_len,
            throughput=output_len/inference_time
        )

    def inference(self):

        # init cache
        self.llm.past_key_values, self.llm.past_key_values_data, self.llm.current_length_data = initialize_past_key_values(self.llm.base_model)
        self.llm.current_length_data.zero_()
        
        # preprocess
        input_ids = self.tokenizer([self.prompt]).input_ids
        input_len = len(input_ids[0])
        
        # inference
        with torch.inference_mode():
            self.llm.current_length_data.zero_()
            start_time = time.time()
            output = self.llm.base_model(torch.as_tensor(input_ids).cuda(), past_key_values=self.llm.past_key_values,)
            output_ids = output.logits.argmax(-1)
            input_ids[0] = input_ids[0] + output_ids[0, -1:].tolist()
            for _ in tqdm(range(self.args.num_out_tokens-1)):
                output = self.llm.base_model(output_ids[..., -1:], past_key_values=self.llm.past_key_values, use_cache=True)
                output_ids = output.logits.argmax(-1)
                # pred_topk = output.logits.topk(10, dim = -1).indices[0]
                input_ids[0] = input_ids[0] + output_ids[0, -1:].tolist()
            end_time = time.time()
        
        # output
        output_ids = input_ids[0][input_len:]
        output_len = len(output_ids)
        output_sentence = self.tokenizer.decode(output_ids)
        inference_time = end_time - start_time
        
        self.logger.info(f"[INPUT] {self.prompt}")
        self.logger.info(f"[OUTPUT] {output_sentence}")
        return input_ids, input_len, output_ids, output_len, output_sentence, inference_time