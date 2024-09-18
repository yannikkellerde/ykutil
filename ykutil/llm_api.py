import json
import os
from typing import Optional

from openai import AzureOpenAI
from openai._base_client import BaseClient


class ModelWrapper:
    def __init__(
        self, client: BaseClient, model_name: str, log_file: Optional[str] = None
    ):
        self.client = client
        self.model_name = model_name
        self.log_file = log_file
        self.stats = {"requests": 0, "input_tokens": 0, "completion_tokens": 0}

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(messages) + ",\n")

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )
        self.stats["requests"] += 1
        self.stats["input_tokens"] += response.usage.prompt_tokens
        self.stats["completion_tokens"] += response.usage.completion_tokens
        return response.choices[0].message.content

    def compute_cost():
        raise NotImplementedError()


class AzureModelWrapper(ModelWrapper):
    def __init__(self, model_name: str = "gpt-4o", log_file=None):
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        super().__init__(self.client, model_name, log_file)

    def compute_cost(
        self,
        input_token_cost: Optional[float] = None,
        output_token_cost: Optional[float] = None,
    ) -> float:
        if input_token_cost is None:
            assert output_token_cost is None
            if "gpt-4o" in self.model_name:
                if "mini" in self.model_name:
                    input_token_cost = 0.000165 / 1000
                    output_token_cost = 0.00066 / 1000
                else:
                    input_token_cost = 0.005 / 1000
                    output_token_cost = 0.015 / 1000
            elif "gpt-35" in self.model_name:
                input_token_cost = 0.0005 / 1000
                output_token_cost = 0.0015 / 1000
            else:
                raise ValueError(
                    f"Unknown model name: {self.model_name}. Please provide"
                    " input_token_cost and output_token_cost."
                )

        total_cost = (
            self.stats["input_tokens"] * input_token_cost
            + self.stats["completion_tokens"] * output_token_cost
        )
        return total_cost
