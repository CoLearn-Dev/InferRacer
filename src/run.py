import time
from typing import Optional
import datetime

from attr import define
import tiktoken
from openai import BaseModel
from transformers import AutoTokenizer

from .workload.workload import ChatCompletion, Workload

llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
openai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class ResponseChunk(BaseModel):
    chunk_id: int
    payload: list[str]
    time_arrived: datetime.datetime


class Summary(BaseModel):
    rule: str
    username: str
    num_token_total_llama3: int
    num_token_total_openai: int
    num_get_total: int
    num_post_total: int
    num_task_finished: int
    num_task_touched: int
    first_get_time: Optional[datetime.datetime]
    last_post_time: Optional[datetime.datetime]
    sample_requests_first_3: list[tuple[ChatCompletion, tuple[str, datetime.datetime]]]
    sample_requests_last_3: list[tuple[ChatCompletion, tuple[str, datetime.datetime]]]
    trace_num_token: list[int]


def count_openai_token(text: str) -> int:
    return len(openai_tokenizer.encode(text))


def count_llama3_token(text: str) -> int:
    return len(llama3_tokenizer.tokenize(text))


class Run:
    """
    Currently, the timer goes when the struct is created.
    """

    def __init__(
        self, username: str, rule: str, workload: Workload, time_limit: int = 60
    ):
        self.username = username
        self.rule = rule
        self.workload = workload
        self.responses: list[ResponseChunk] = []
        self.time_started = datetime.datetime.utcnow()
        self.time_limit = time_limit
        self.num_get_total = 0
        self.num_post_total = 0
        self.time_first_get: Optional[datetime.datetime] = None
        self.trace_num_token: list[int] = []

    def get_workload(self, offset: int, limit: int) -> Optional[list[ChatCompletion]]:
        now = datetime.datetime.utcnow()
        if now - self.time_started > datetime.timedelta(seconds=self.time_limit):
            return None
        now = datetime.datetime.utcnow()
        if self.num_get_total == 0:
            self.time_first_get = now
        self.time_last_get = now
        self.num_get_total += 1
        return self.workload.completions[offset : offset + limit]

    def add_result(
        self, chunk_id: int, offset: int, data: list[str], finished: bool
    ) -> bool:
        now = datetime.datetime.utcnow()
        if now - self.time_started > datetime.timedelta(seconds=self.time_limit):
            return False

        self.num_post_total += 1

        if finished:
            self.time_consumed = now - self.time_started
            return True

        flag = False
        for index, response in enumerate(self.responses):
            if response.chunk_id > chunk_id:
                self.responses.insert(
                    index,
                    ResponseChunk(chunk_id=chunk_id, payload=data, time_arrived=now),
                )
                flag = True
        if not flag:
            self.responses.append(
                ResponseChunk(chunk_id=chunk_id, payload=data, time_arrived=now)
            )

        return True

    def get_timestamped_response(self) -> list[tuple[str, datetime.datetime]]:
        responses = []
        for chunk in self.responses:
            for entry in chunk.payload:
                responses.append((entry, chunk.time_arrived))
        return responses

    def calculate_trace(self):
        trace = []
        responses = list(
            map(
                lambda x: (count_openai_token(x[0]), x[1]),
                self.get_timestamped_response(),
            )
        )

        time_first_get = self.time_first_get
        assert time_first_get is not None

        for i in range(1, self.time_limit + 1):
            satisfied_num = sum(
                map(
                    lambda x: x[0],
                    filter(
                        lambda x: x[1] - time_first_get
                        <= datetime.timedelta(seconds=i),
                        responses,
                    ),
                )
            )
            trace.append(satisfied_num)
        return trace

    def sumup(self) -> Summary:
        if len(self.responses) > 0:
            last_post_time = self.responses[-1].time_arrived
        responses = self.get_timestamped_response()

        return Summary(
            rule=self.rule,
            username=self.username,
            num_token_total_llama3=sum(
                count_llama3_token(entry[0]) for entry in responses
            ),
            num_token_total_openai=sum(
                count_openai_token(entry[0]) for entry in responses
            ),
            num_get_total=self.num_get_total,
            num_post_total=self.num_post_total,
            num_task_finished=self.num_post_total,
            num_task_touched=self.num_get_total,
            first_get_time=self.time_first_get,
            last_post_time=last_post_time,
            sample_requests_first_3=list(
                zip(self.workload.completions[:3], responses[:3])
            ),
            sample_requests_last_3=list(
                zip(self.workload.completions[-3:], responses[-3:])
            ),
            trace_num_token=self.calculate_trace(),
        )
