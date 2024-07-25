from typing import Optional
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request
from transformers import AutoTokenizer
import mysql.connector
from fleece_network.sede import dumps
import multiprocessing
import uuid

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
stop_tokens = {128001, 128009}

import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def get_worker_ids():
    password = os.environ.get("DB_PASSWORD")
    if password is None:
        print("Please set DB_PASSWORD environment variable")
        exit(1)

    mydb = mysql.connector.connect(
        host="127.0.0.1",
        user="fleece",
        password=password,
        database="fleece",
    )

    mycursor = mydb.cursor()

    mycursor.execute(
        "SELECT id FROM worker WHERE nickname = 'llama3' AND status = 'online'"
    )

    myresult = mycursor.fetchall()

    print("Worker IDs:")
    for x in myresult:
        print(x[0])

    return myresult


worker_ids = get_worker_ids()

if len(worker_ids) == 0:
    print("No workers available")
    exit(1)
elif len(worker_ids) == 1:
    print("Benchmark will run llama-3-8b on a single worker")
    plan_template = [
        [
            worker_ids[0][0],
            [
                "llama-3-8b-instruct-slice/tok_embeddings",
                *[f"llama-3-8b-instruct-slice/layers.{i}" for i in range(32)],
                "llama-3-8b-instruct-slice/norm",
                "llama-3-8b-instruct-slice/output",
            ],
        ]
    ]
elif len(worker_ids) == 2:
    plan_template = [
        [
            worker_ids[0][0],
            [
                "llama-3-70b-instruct-slice/tok_embeddings",
                *[f"llama-3-70b-instruct-slice/layers.{i}" for i in range(0, 40)],
            ],
        ],
        [
            worker_ids[1][0],
            [
                *[f"llama-3-70b-instruct-slice/layers.{i}" for i in range(40, 80)],
                "llama-3-70b-instruct-slice/norm",
                "llama-3-70b-instruct-slice/output",
            ],
        ],
    ]


class Client:
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        batchsize: int,
    ):
        self.username = username
        self.password = password
        self.url = url
        self.batchsize = batchsize
        self.vllm_executor = ThreadPoolExecutor(max_workers=16)
        self.network_executor = ThreadPoolExecutor(max_workers=16)

        requests.post(
            f"{url}/user/signin",
            data={"username": username, "password": password},
        )

        token = requests.post(
            f"{url}/user/login",
            data={"username": username, "password": password},
        ).json()["token"]
        self.header = {"Authorization": f"Bearer {token}"}
        self.queue = multiprocessing.Queue()

    def flask_service(self, events: multiprocessing.Queue, run_id: str):
        app = Flask(__name__)

        @app.route("/update_tasks", methods=["POST"])
        def update_tasks():
            data = request.json
            entries = []
            for task in data:
                request_id = int(task["task_id"].split("@")[0])
                token = task["output_tokens"][0][0]
                char = tokenizer.decode(task["output_tokens"][0][0])
                entries.append(
                    {
                        "request_id": request_id,
                        "offset": -1,
                        "payload": char,
                        "finished": False,
                    }
                )
                if token in stop_tokens:
                    events.put(request_id)
            print(len(entries))
            if not self.send_responses(run_id, entries):
                events.put("stop")

            return "OK"

        app.run(port=29980, debug=False)

    def recv_prompts(self, run_id: str, offset: int, limit: int) -> Optional[dict]:
        reply = requests.get(
            f"{self.url}/run/lt/{run_id}/fetch",
            headers=self.header,
            params={"offset": offset, "limit": limit, "sorted": False},
        )
        if reply.status_code != 200:
            return None
        return reply.json()

    def send_responses(self, run_id: str, entries: list[dict]) -> bool:
        reply = requests.post(
            f"{self.url}/run/lt/{run_id}/submit",
            headers=self.header,
            json=entries,
        )
        return reply.status_code == 200

    def sumup(self, run_id: str):
        reply = requests.get(f"{self.url}/run/lt/{run_id}/summary", headers=self.header)
        print(json.dumps(reply.json()))
        reply = requests.get(f"{self.url}/run/lt/leaderboard", headers=self.header)
        print(json.dumps(reply.json()))

    def speedtest(self):
        reply = requests.post(
            f"{self.url}/run/lt",
            headers=self.header,
            json={"time_limit": 60, "shuffled": False, "model": "llama"},
        )
        run_id = reply.json()["run_id"]
        p = multiprocessing.Process(
            target=self.flask_service, args=[self.queue, run_id]
        )
        p.start()

        try:
            prefetch_future = self.network_executor.submit(
                self.recv_prompts,
                run_id,
                -1,
                1,
            )
            current = 0
            while True:
                if current >= self.batchsize:
                    value = self.queue.get()
                    if value == "stop":
                        break
                else:
                    current += 1
                completions = prefetch_future.result()
                if completions is None:
                    break
                prefetch_future = self.network_executor.submit(
                    self.recv_prompts, run_id, -1, 1
                )
                messages = completions[0]["payload"]["messages"]
                request_id = completions[0]["request_id"]
                formatted_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
                metadata = {
                    "task_id": str(request_id) + "@" + str(uuid.uuid4()),
                    "step": 0,
                    "round": 0,
                    "plan": plan_template,
                    "payload": [formatted_tokens],
                    "task_manager_url": "http://localhost:29980",
                    "max_total_len": 2048,
                }
                response = requests.post(
                    "http://localhost:8080/forward", data=dumps({}, metadata)
                )
            self.sumup(run_id)
            p.kill()
        except KeyboardInterrupt:
            self.sumup(run_id)
            p.kill()


if __name__ == "__main__":
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if username is None or password is None:
        print("Please set USERNAME and PASSWORD environment variables")
        exit(1)
    cli = Client(username, password, "http://localhost:28000", 66)
    cli.speedtest()
