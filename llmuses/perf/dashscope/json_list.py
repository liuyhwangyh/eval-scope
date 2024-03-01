
from sys import maxsize
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
                      min_length: int = 0, **kwargs: Any) -> Iterator[Dict]:
        """Read dataset and return prompt.
           Datasets: https://huggingface.co/datasets/Yukang/LongAlpaca-12k
        """
        if prompt is not None:
            messages = [{'role': 'user', 'content': prompt}]
            yield {
                "model": model,
                "input": {"messages": messages},
                "parameters": {"stream": True,
                               "incremental_output": True,
                               **kwargs}
            }
        elif dataset is not None:
            with open(dataset, 'r', encoding='utf-8') as f:
                content = f.read()
            data = json.loads(content)
            for _, item in enumerate(data):
                instruction = item['instruction']
                if len(instruction) > min_length and len(instruction) < max_length:
                    messages = [{'role': 'user', 'content': instruction}]
                    yield {
                        "model": model,
                        "input": {"messages": messages},
                        "parameters": {"stream": True,
                                       "incremental_output": True,
                                       **kwargs}
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
        return js['usage']['input_tokens'], js['usage']['output_tokens']
