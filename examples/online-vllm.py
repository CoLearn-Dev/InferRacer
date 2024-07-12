from typing import Optional
from vllm import LLM
import aiohttp
import requests
import json
import os
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from openai import OpenAI
import time
import asyncio

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


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
        self.vllm_executor = ThreadPoolExecutor(max_workers=16)
        self.network_executor = ThreadPoolExecutor(max_workers=16)
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

    def recv_prompts(self, run_id: int, offset: int, limit: int) -> Optional[dict]:
        reply = requests.get(
            f"{self.url}/run/lt/{run_id}/fetch",
            headers=self.header,
            params={"offset": offset, "limit": limit, "sorted": False},
        )
        if reply.status_code != 200:
            return None
        return reply.json()

    def send_responses(self, run_id: int, entries: list[dict]) -> bool:
        reply = requests.post(
            f"{self.url}/run/lt/{run_id}/submit",
            headers=self.header,
            json=entries,
        )
        return reply.status_code == 200

    def gen_chat_responses(self, request_id: int, messages: list):
        chat_stream = self.openai_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
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

    def relay_responses(self, run_id: int, request_id: int, messages: list):
        chat_stream = self.openai_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
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

    def isolated_speedtest(self, run_id: int):
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
            messages = completions[0]["payload"]["messages"]
            request_id = completions[0]["request_id"]
            if self.mode == "batched":
                chat_response = self.openai_client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=messages,
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

    def speedtest(self):
        reply = requests.post(
            f"{self.url}/run/lt",
            headers=self.header,
            json={"rule": "60s", "model": "llama"},
        )
        run_id = reply.json()["run_id"]
        futures = []
        for _ in range(self.batchsize):
            futures.append(self.vllm_executor.submit(self.isolated_speedtest, run_id))

        _, _ = wait(futures, return_when=ALL_COMPLETED)

        reply = requests.get(f"{self.url}/run/lt/{run_id}/summary", headers=self.header)
        print(json.dumps(reply.json()))
        reply = requests.get(f"{self.url}/run/lt/leaderboard", headers=self.header)
        print(json.dumps(reply.json()))


if __name__ == "__main__":
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if username is None or password is None:
        print("Please set USERNAME and PASSWORD environment variables")
        exit(1)
    cli = Client(username, password, "http://localhost:28000", 128, "stream")
    cli.speedtest()
