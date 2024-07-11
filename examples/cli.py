from typing import Optional
from vllm import LLM
import requests
import json
from transformers import AutoTokenizer
import os
from concurrent.futures import ThreadPoolExecutor
import time


class Client:
    def __init__(self, username: str, password: str, url: str, batchsize: int):
        self.username = username
        self.password = password
        self.url = url
        self.batchsize = batchsize
        self.executor = ThreadPoolExecutor(max_workers=16)

        requests.post(
            f"{url}/user/signin",
            data={"username": username, "password": password},
        )

        token = requests.post(
            f"{url}/user/login",
            data={"username": username, "password": password},
        ).json()["token"]
        self.header = {"Authorization": f"Bearer {token}"}

    def recv_prompts(self, run_id: int, offset: int, limit: int) -> Optional[dict]:
        reply = requests.get(
            f"{self.url}/run/lt/{run_id}/batch",
            headers=self.header,
            params={"offset": offset, "limit": limit, "sorted": False},
        )
        if reply.status_code != 200:
            return None
        return reply.json()

    def send_responses(self, run_id: int, entries: list[dict]) -> bool:
        reply = requests.post(
            f"{self.url}/run/lt/{run_id}",
            headers=self.header,
            json={
                "entries": entries,
            },
        )
        return reply.status_code == 200

    def speedtest(self, llm: LLM):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        reply = requests.post(
            f"{self.url}/run/lt",
            headers=self.header,
            json={"rule": "60s", "model": "llama"},
        )
        run_id = reply.json()["run_id"]

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
                    completion["messages"], tokenize=False, add_generation_prompt=True
                )
                prompts.append(formatted_prompt)
            future = self.executor.submit(llm.generate, prompts)
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

        reply = requests.get(f"{self.url}/run/lt/{run_id}/summary", headers=self.header)
        print(json.dumps(reply.json()))


if __name__ == "__main__":
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if username is None or password is None:
        print("Please set USERNAME and PASSWORD environment variables")
        exit(1)
    cli = Client(username, password, "http://localhost:28000", 128)
    llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
    cli.speedtest(llm)
