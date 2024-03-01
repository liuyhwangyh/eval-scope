
import sys
from typing import Any, Dict, Iterator
import json
from ..llm_parser_base import PerfPluginBase


class PerfPlugin(PerfPluginBase):
    def build_request(self,
                      model: str,
                      prompt: str = None,
                      dataset: str = None,
                      max_length: int = sys.maxsize,
                      min_length: int = 0,
                      **kwargs: Any) -> Iterator[Dict]:
        if prompt is not None:
            yield {
                "model": model,
                "prompt": "<|im_start|>system\nYour are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % prompt,
                "stop": ["<|im_end|>", "<|endoftext|>"],
                "stream": True,
                **kwargs
            }
        elif dataset is not None:
            for item in self.dataset_line_by_line(dataset):
                instruction = item['instruction']
                if len(instruction) > min_length and len(instruction) < max_length:
                    yield {
                        "model": model,
                        "prompt": "<|im_start|>system\nYour are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % prompt,
                        "stop": ["<|im_end|>", "<|endoftext|>"],
                        "stream": True,
                        **kwargs
                    }
        else:
            raise Exception('prompt or dataset is required!')

    def parse_responses(self, responses, **kwargs) -> Dict:
        """Parser responses and return number of request and response tokens.

        Args:
            responses (List[bytes]): List of http response body, for stream output,
                there are multiple responses, for general only one. 
            kwargs: (Any): The command line --parameter content.

        Returns:
            Tuple: Return number of prompt token and number of completion tokens.
        """
        last_response = responses[-1]
        js = json.loads(last_response)
        return js['usage']['prompt_tokens'], js['usage']['completion_tokens']
