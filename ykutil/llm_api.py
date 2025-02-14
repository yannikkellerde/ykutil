import requests
import os
import json
import argparse


def process_file(
    file_path: str,
    model_name: str,
    url: str = "http://localhost:30000/v1/chat/completions",
    temperature: float = 1.0,
    max_tokens: int = 100,
):

    with open(file_path, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    for example in data:
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
