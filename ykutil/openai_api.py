import base64
import time
import datetime
from copy import deepcopy
import json
import os
from mimetypes import guess_type
from typing import Optional, Type

from openai import AzureOpenAI
from openai._base_client import BaseClient

from ykutil.types_util import T
from ykutil.log_util import log


# Source: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest
# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def human_readable_parse(messages: list[dict[str, str]]):
    return "\n".join([f'{msg["role"]}:\n{msg["content"]}' for msg in messages])


class ModelWrapper:
    def __init__(
        self, client: BaseClient, model_name: str, log_file: Optional[str] = None
    ):
        self.client = client
        self.model_name = model_name
        self.log_file = log_file
        self.stats = {"requests": 0, "input_tokens": 0, "completion_tokens": 0}

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )
        self.stats["requests"] += 1
        self.stats["input_tokens"] += response.usage.prompt_tokens
        self.stats["completion_tokens"] += response.usage.completion_tokens
        if self.log_file is not None:
            msg_copy = messages.copy()
            msg_copy.append(response.choices[0].message.dict())
            with open(self.log_file, "a") as f:
                f.write(json.dumps(msg_copy) + ",\n")
        out = response.choices[0].message.content
        if out is None:
            log("Bad response", response, level="warn")
        return out

    def structured_complete(
        self, messages: list[dict[str, str]], structure_class: Type[T], **kwargs
    ):
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=structure_class,
        )
        self.stats["requests"] += 1
        self.stats["input_tokens"] += response.usage.prompt_tokens
        self.stats["completion_tokens"] += response.usage.completion_tokens
        if self.log_file is not None:
            msg_copy = messages.copy()
            msg_copy.append(response.choices[0].message.dict())
            with open(self.log_file, "a") as f:
                f.write(json.dumps(msg_copy) + ",\n")

        out = response.choices[0].message
        if out is None:
            log("Bad response", response, level="warn")
        return out

    def compute_cost():
        raise NotImplementedError()


class AzureModelWrapper(ModelWrapper):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        log_file=None,
        api_version="2024-08-01-preview",
    ):
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
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
                    if "2024-08-06" in self.model_name:
                        input_token_cost = 0.0025 / 1000
                        output_token_cost = 0.010 / 1000
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


class BatchAPIWrapper:
    def __init__(self, model_name, api_version="2024-08-01-preview", tmp_dir="~/tmp"):
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.model_name = model_name
        self.job_json = {
            "custom_id": None,
            "method": "POST",
            "url": "/chat/completions",
            "body": {"model": self.model_name, "messages": None},
        }
        self.temp_dir = os.path.expanduser(tmp_dir)

    def create_batch_job(self, many_messages: list[list[dict[str, str]]]):
        with open(os.path.join(self.temp_dir, "batch_job.jsonl"), "w") as f:
            for i, messages in enumerate(many_messages):
                job = deepcopy(self.job_json)
                job["body"]["messages"] = messages
                job["custom_id"] = str(i)
                f.write(json.dumps(job) + "\n")

        file = self.client.files.create(file=open("test.jsonl", "rb"), purpose="batch")
        batch_response = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/chat/completions",
            completion_window="24h",
        )

        return batch_response.id

    def get_result(self, batch_id):
        status = "validating"
        while status not in ("completed", "failed", "canceled"):
            time.sleep(60)
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status
            print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

        if batch_response.status == "failed":
            for error in batch_response.errors.data:
                print(f"Error code {error.code} Message {error.message}")
        output_file_id = batch_response.output_file_id

        if not output_file_id:
            output_file_id = batch_response.error_file_id

        if output_file_id:
            file_response = self.client.files.content(output_file_id)
            raw_responses = file_response.text.strip().split("\n")

            for raw_response in raw_responses:
                json_response = json.loads(raw_response)
                formatted_json = json.dumps(json_response, indent=2)
                print(formatted_json)
