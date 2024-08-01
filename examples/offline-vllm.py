import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import requests
from transformers import AutoTokenizer  # type: ignore
from vllm import LLM, SamplingParams


class Client:
    def __init__(self, username: str, password: str, url: str, batchsize: int):
        self.username = username
        self.password = password
        self.url = url
        self.batchsize = batchsize
        self.executor = ThreadPoolExecutor(max_workers=128)

        requests.post(
            f"{url}/user/signin",
            data={"username": username, "password": password},
        )

        token = requests.post(
            f"{url}/user/login",
            data={"username": username, "password": password},
        ).json()["token"]
        self.header = {"Authorization": f"Bearer {token}"}

    def recv_prompts(
        self, run_id: int, offset: int, limit: int
    ) -> Optional[list[dict[Any, Any]]]:
        reply = requests.get(
            f"{self.url}/run/lt/{run_id}/fetch",
            headers=self.header,
            params={"offset": offset, "limit": limit, "sorted": False},
        )
        if reply.status_code != 200:
            return None
        return reply.json()

    def send_responses(self, run_id: int, entries: list[dict[Any, Any]]) -> bool:
        reply = requests.post(
            f"{self.url}/run/lt/{run_id}/submit",
            headers=self.header,
            json=entries,
        )
        return reply.status_code == 200

    def sumup(self, run_id: str) -> None:
        reply = requests.get(f"{self.url}/run/lt/{run_id}/summary", headers=self.header)
        print(json.dumps(reply.json()))
        reply = requests.get(f"{self.url}/run/lt/leaderboard", headers=self.header)
        print(json.dumps(reply.json()))

    def speedtest(self, llm: LLM, time_limit: int, max_tokens: int) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-70B-Instruct"
        )
        reply = requests.post(
            f"{self.url}/run/lt",
            headers=self.header,
            json={"time_limit": time_limit, "shuffled": False, "model": "llama"},
        )
        run_id = reply.json()["run_id"]

        try:
            index = 0
            prefetch_future = self.executor.submit(
                self.recv_prompts, run_id, 0, self.batchsize
            )
            while True:
                completions = prefetch_future.result()
                if completions is None:
                    break
                prefetch_future = self.executor.submit(
                    self.recv_prompts, run_id, index + self.batchsize, self.batchsize
                )
                prompts = []
                for completion in completions:
                    formatted_prompt = tokenizer.apply_chat_template(
                        completion["payload"]["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompts.append(formatted_prompt)
                samparam = SamplingParams(max_tokens=max_tokens)
                future = self.executor.submit(llm.generate, prompts, samparam)
                outputs = future.result()
                results = [output.outputs[0].text for output in outputs]
                entries = []
                for result in results:
                    entries.append(
                        {
                            "request_id": index,
                            "offset": 0,
                            "payload": result,
                            "finished": False,
                        }
                    )
                    index += 1
                self.executor.submit(self.send_responses, run_id, entries)
            self.sumup(run_id)
        except KeyboardInterrupt:
            self.sumup(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct"
    )
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()

    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if username is None or password is None:
        print("Please set USERNAME and PASSWORD environment variables")
        exit(1)
    cli = Client(username, password, "http://localhost:28000", args.batch_size)
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=0.95,
    )
    cli.speedtest(llm, args.time_limit, args.max_tokens)
