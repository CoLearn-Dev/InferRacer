import argparse
import asyncio
import json
import os
import time
import traceback
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Generator, Optional

import aiohttp
import requests
from openai import OpenAI
from transformers import AutoTokenizer  # type: ignore
from vllm import LLM

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8081/v1"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


class Client:
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        batchsize: int,
        mode: str = "batched",
    ):
        self.username = username
        self.password = password
        self.url = url
        self.batchsize = batchsize
        self.vllm_executor = ThreadPoolExecutor(max_workers=128)
        self.network_executor = ThreadPoolExecutor(max_workers=128)
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.mode = mode

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
    ) -> Optional[dict[Any, Any]]:
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

    def gen_chat_responses(self, request_id: int, messages: list) -> Generator[bytes]:
        chat_stream = self.openai_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=messages,
            stream=True,
        )
        for item in chat_stream:
            choice = item.choices[0]
            content = choice.delta.content
            if content == None:
                continue

            entries = [
                {
                    "request_id": request_id,
                    "offset": -1,
                    "payload": content,
                    "finished": False,
                }
            ]
            yield json.dumps(entries).encode("utf-8") + b"\n"

    def resolve_responses_streamed(
        self, run_id: int, request_id: int, messages: list
    ) -> bool:
        reply = requests.post(
            f"{self.url}/run/lt/{run_id}/push",
            headers=self.header,
            data=self.gen_chat_responses(request_id, messages),
            stream=True,
        )
        return reply.status_code != 200

    def relay_responses(self, run_id: int, request_id: int, messages: list) -> None:
        chat_stream = self.openai_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=messages,
            stream=True,
        )
        for item in chat_stream:
            choice = item.choices[0]
            content = choice.delta.content
            if content == None:
                continue

            entries = [
                {
                    "request_id": request_id,
                    "offset": -1,
                    "payload": content,
                    "finished": False,
                }
            ]
            self.network_executor.submit(self.send_responses, run_id, entries)

    def isolated_speedtest(self, run_id: int, max_tokens: int) -> None:
        try:
            prefetch_future = self.network_executor.submit(
                self.recv_prompts,
                run_id,
                -1,
                1,
            )
            while True:
                completions = prefetch_future.result()
                if completions is None:
                    break
                prefetch_future = self.network_executor.submit(
                    self.recv_prompts, run_id, -1, 1
                )
                request_id = completions[0]["request_id"]
                messages = completions[0]["payload"]["messages"]
                prompts = []
                for completion in completions:
                    formatted_prompt = tokenizer.apply_chat_template(
                        completion["payload"]["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompts.append(formatted_prompt)
                if self.mode == "batched":
                    chat_response = self.openai_client.completions.create(
                        model="meta-llama/Meta-Llama-3-70B-Instruct",
                        prompt=prompts,
                        max_tokens=max_tokens,
                    )
                    entries = [
                        {
                            "request_id": request_id,
                            "offset": 0,
                            "payload": chat_response.choices[0].message.content,
                            "finished": False,
                        }
                    ]
                    self.network_executor.submit(self.send_responses, run_id, entries)
                else:
                    self.resolve_responses_streamed(run_id, request_id, messages)
        except Exception as e:
            traceback.print_exc()

    def sumup(self, run_id: str) -> None:
        reply = requests.get(f"{self.url}/run/lt/{run_id}/summary", headers=self.header)
        print(json.dumps(reply.json()))
        reply = requests.get(f"{self.url}/run/lt/leaderboard", headers=self.header)
        print(json.dumps(reply.json()))

    def speedtest(self, time_limit: int, max_tokens: int) -> None:
        reply = requests.post(
            f"{self.url}/run/lt",
            headers=self.header,
            json={"time_limit": time_limit, "shuffled": False, "model": "llama"},
        )
        run_id = reply.json()["run_id"]
        futures = []
        try:
            for _ in range(self.batchsize):
                futures.append(
                    self.vllm_executor.submit(
                        self.isolated_speedtest, run_id, max_tokens
                    )
                )

            _, _ = wait(futures, return_when=ALL_COMPLETED)
            self.sumup(run_id)
        except KeyboardInterrupt:
            self.sumup(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if username is None or password is None:
        print("Please set USERNAME and PASSWORD environment variables")
        exit(1)
    cli = Client(
        username, password, "http://localhost:28000", args.batch_size, "stream"
    )
    cli.speedtest(args.time_limit, args.max_tokens)
