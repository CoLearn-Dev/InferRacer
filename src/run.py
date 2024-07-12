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


class BufferedString(BaseModel):
    buffer: str = ""
    time_last_added: Optional[datetime.datetime] = None
    finished: bool = False

    def insert(
        self, offset: int, payload: str, time: datetime.datetime, finished: bool
    ):
        self.time_last_added = time
        self.finished = finished
        if offset == -1:
            self.buffer = self.buffer + payload
        elif offset <= len(self.buffer):
            self.buffer = self.buffer[:offset] + payload
        else:
            self.buffer = self.buffer + " " * (offset - len(self.buffer)) + payload


class BufferedResponses(BaseModel):
    content: list[BufferedString] = []

    def insert(
        self,
        index: int,
        offset: int,
        payload: str,
        time: datetime.datetime,
        finished: bool,
    ):
        while index >= len(self.content):
            self.content.append(BufferedString())
        self.content[index].insert(offset, payload, time, finished)

    def into_timestamped_responses(
        self,
    ) -> list[tuple[str, Optional[datetime.datetime]]]:
        return list(map(lambda x: (x.buffer, x.time_last_added), self.content))


class RequestEntry(BaseModel):
    request_id: int
    payload: ChatCompletion


class RequestChunk(BaseModel):
    entries: list[RequestEntry]


class ResponseEntry(BaseModel):
    request_id: int
    offset: int
    payload: str
    finished: bool


class ResponseChunk(BaseModel):
    entries: list[ResponseEntry]


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
    sample_requests_first_3: list[tuple[ChatCompletion, BufferedString]]
    sample_requests_last_3: list[tuple[ChatCompletion, BufferedString]]
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
        self.responses = BufferedResponses()
        self.time_started = datetime.datetime.utcnow()
        self.time_limit = time_limit
        self.num_get_total = 0
        self.num_post_total = 0
        self.time_first_get: Optional[datetime.datetime] = None
        self.trace_num_token: list[int] = []
        self.pointer = 0

    def get_workload(self, offset: int, limit: int) -> Optional[RequestChunk]:
        now = datetime.datetime.utcnow()
        if now - self.time_started > datetime.timedelta(seconds=self.time_limit):
            return None
        now = datetime.datetime.utcnow()
        if self.num_get_total == 0:
            self.time_first_get = now
        self.time_last_get = now
        self.num_get_total += 1

        entries = []
        if offset == -1:
            offset = self.pointer
        for completion, index in zip(
            self.workload.completions[offset : offset + limit],
            range(offset, offset + limit),
        ):
            entries.append(RequestEntry(request_id=index, payload=completion))
        self.pointer = max(self.pointer, offset + limit)
        return RequestChunk(entries=entries)

    def add_result(
        self,
        responses: list[ResponseEntry],
    ) -> bool:
        now = datetime.datetime.utcnow()
        if now - self.time_started > datetime.timedelta(seconds=self.time_limit):
            return False

        self.num_post_total += 1

        for entry in responses:
            self.responses.insert(
                entry.request_id, entry.offset, entry.payload, now, False
            )

        return True

    def calculate_trace(self):
        trace = []
        responses = list(
            map(
                lambda x: (count_openai_token(x.buffer), x.time_last_added),
                filter(lambda x: x.time_last_added is not None, self.responses.content),
            )
        )

        time_first_get = self.time_first_get
        assert time_first_get is not None

        for i in range(1, self.time_limit + 1):
            satisfied_num = sum(
                map(
                    lambda x: x[0],
                    filter(
                        lambda x: x[1] - time_first_get  # type: ignore
                        <= datetime.timedelta(seconds=i),
                        responses,
                    ),
                )
            )
            trace.append(satisfied_num)
        return trace

    def sumup(self) -> Summary:
        if len(self.responses.content) > 0:
            last_post_time = self.responses.content[-1].time_last_added
        else: 
            last_post_time = None

        return Summary(
            rule=self.rule,
            username=self.username,
            num_token_total_llama3=sum(
                count_llama3_token(entry.buffer) for entry in self.responses.content
            ),
            num_token_total_openai=sum(
                count_openai_token(entry.buffer) for entry in self.responses.content
            ),
            num_get_total=self.num_get_total,
            num_post_total=self.num_post_total,
            num_task_finished=self.num_post_total,
            num_task_touched=self.num_get_total,
            first_get_time=self.time_first_get,
            last_post_time=last_post_time,
            sample_requests_first_3=list(
                zip(self.workload.completions[:3], self.responses.content[:3])
            ),
            sample_requests_last_3=list(
                zip(self.workload.completions[-3:], self.responses.content[-3:])
            ),
            trace_num_token=self.calculate_trace(),
        )
