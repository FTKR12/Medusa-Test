import json
import logging
from typing import Dict, TypedDict

class LLMOutput(TypedDict):
    model_name: str
    input_tokens: int
    inference_time: float
    output_tokens: int
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
    
    @abstractmethod
    def run(self) -> Dict[]:
        pass

class DefaultLLMRunner():
    def __init__(self):
        pass

    def run(self):
        pass

class MedusaRunner():
    def __init__(self):
        pass

    def run(self):
        pass