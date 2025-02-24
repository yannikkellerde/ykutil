import requests
import os
import json
import argparse
from urllib.parse import urljoin


class SglangModelWrapper:
    def __init__(self, host: str, model_name: str):
        self.host = host
        self.model_name = model_name

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = requests.post(
            urljoin(self.host, "/v1/chat/completions"),
            json={"model": self.model_name, "messages": messages, **kwargs},
        )
        return response.json()["choices"][0]["message"]["content"]


def process_file(
    file_path: str,
    model_name: str,
    url: str = "http://localhost:30000/v1/chat/completions",
    temperature: float = 1.0,
    max_tokens: int = 100,
):

    with open(file_path, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    made_changes = False
    for example in data:
        if "response" not in example:
            made_changes = True
            llm_request = {
                "model": model_name,
                "messages": [{"role": "user", "content": example["prompt"]}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            header = {
                "Content-Type": "application/json",
            }
            result = requests.post(url, json=llm_request, headers=header)
            example["response"] = result.json()["choices"][0]["message"]["content"]

    if made_changes:
        with open(file_path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")


def process_folder(
    folder_path: str,
    model_name: str,
    url: str = "http://localhost:30000/v1/chat/completions",
    temperature: float = 1.0,
    max_tokens: int = 100,
):
    for file in os.listdir(folder_path):
        if file.endswith(".jsonl"):
            process_file(
                os.path.join(folder_path, file),
                model_name,
                url,
                temperature,
                max_tokens,
            )


def count_fulfilled_requests(
    folder_path: str,
):
    fulfilled = 0
    todo = 0
    for file in os.listdir(folder_path):
        if file.endswith(".jsonl"):
            with open(os.path.join(folder_path, file), "r") as f:
                data = [json.loads(x) for x in f.readlines()]
            for example in data:
                if "response" in example:
                    fulfilled += 1
                else:
                    todo += 1
    print(f"Fulfilled: {fulfilled}, Todo: {todo}")


def do_process_folder():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument(
        "--url", type=str, default="http://localhost:30000/v1/chat/completions"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()
    process_folder(
        args.folder_path, args.model_name, args.url, args.temperature, args.max_tokens
    )


def do_process_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument(
        "--url", type=str, default="http://localhost:30000/v1/chat/completions"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()
    process_file(
        args.file_path, args.model_name, args.url, args.temperature, args.max_tokens
    )
