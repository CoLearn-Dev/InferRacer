from vllm import LLM
import requests
import json
from transformers import AutoTokenizer
import os


class Client:
    def __init__(self, username: str, password: str, url: str, batchsize: int):
        self.username = username
        self.password = password
        self.url = url
        self.batchsize = batchsize

        requests.post(
            f"{url}/user/signin",
            data={"username": "admin", "password": "admin"},
        )

        token = requests.post(
            f"{url}/user/login",
            data={"username": "admin", "password": "admin"},
        ).json()["token"]
        self.header = {"Authorization": f"Bearer {token}"}

    def speedtest(self, llm: LLM):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        reply = requests.post(
            f"{self.url}/run/lt",
            headers=self.header,
            json={"rule": "60s", "model": "llama"},
        )
        run_id = reply.json()["run_id"]

        index = 0
        chunk_id = 0
        while True:
            reply = requests.get(
                f"{self.url}/run/lt/{run_id}/batch",
                headers=self.header,
                params={"offset": index, "limit": self.batchsize, "sorted": False},
            )
            if reply.status_code != 200:
                break

            completions = reply.json()
            prompts = []
            for completion in completions:
                formatted_prompt = tokenizer.apply_chat_template(
                    completion["messages"], tokenize=False, add_generation_prompt=True
                )
                prompts.append(formatted_prompt)
            outputs = llm.generate(prompts)
            results = [output.outputs[0].text for output in outputs]
            reply = requests.post(
                f"{self.url}/run/lt/{run_id}",
                headers=self.header,
                json={
                    "offset": index,
                    "chunk_id": chunk_id,
                    "data": results,
                    "finished": False,
                },
            )
            if reply.status_code != 200:
                break

            index += self.batchsize
            chunk_id += 1

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
