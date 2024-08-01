import datetime
import pickle
import time
from abc import ABC, abstractmethod
from typing import Optional

import tiktoken
from openai import BaseModel
from transformers import AutoTokenizer  # type: ignore

from .workload.workload import ChatCompletion, Workload

llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
openai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class BufferedString(BaseModel):
    buffer: str = ""
    time_last_added: Optional[datetime.datetime] = None
    finished: bool = False

    def insert(
        self, offset: int, payload: str, time: datetime.datetime, finished: bool
    ) -> None:
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
    ) -> None:
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


class ResponseEntry(BaseModel):
    request_id: int
    offset: int
    payload: str
    finished: bool


class Summary(BaseModel):
    run_id: str
    username: str
    time_used: float
    num_token_total_llama3_avg: float
    num_token_total_openai_avg: float
    num_get_total: int
    num_post_total: int
    num_task_finished: int
    num_task_touched: int
    first_get_time: Optional[datetime.datetime]
    last_post_time: Optional[datetime.datetime]
    sample_requests_first_3: list[tuple[ChatCompletion, BufferedString]]
    sample_requests_last_3: list[tuple[ChatCompletion, BufferedString]]
    trace_num_token: list[int]
    better_trace: dict[int, int]


def count_openai_token(text: str) -> int:
    return len(openai_tokenizer.encode(text))


def count_llama3_token(text: str) -> int:
    return len(llama3_tokenizer.tokenize(text))


class BasicRun(ABC):
    def __init__(self, run_id: str, username: str, workload: Workload):
        self.run_id = run_id
        self.username = username
        self.workload = workload
        self.responses = BufferedResponses()
        self.num_get_total = 0
        self.num_post_total = 0  # meaningless when using streaming?
        self.workload_index = 0
        self.time_started = datetime.datetime.utcnow()
        self.time_last_added: Optional[datetime.datetime] = None
        self.trace: dict[int, int] = {}

    @abstractmethod
    def check_limitation(self) -> bool:
        pass

    def get_time_used(self) -> Optional[float]:
        if self.time_last_added is None:
            return None
        else:
            return (self.time_last_added - self.time_started).total_seconds()

    def get_workload(self, offset: int, limit: int) -> Optional[list[RequestEntry]]:
        if not self.check_limitation():
            return None

        self.num_get_total += 1

        entries = []
        if offset == -1:
            offset = self.workload_index
        for completion, index in zip(
            self.workload.completions[offset : offset + limit],
            range(offset, offset + limit),
        ):
            entries.append(RequestEntry(request_id=index, payload=completion))
        self.workload_index = max(self.workload_index, offset + limit)
        return entries

    def add_result(
        self,
        responses: list[ResponseEntry],
    ) -> bool:
        if not self.check_limitation():
            return False

        self.num_post_total += 1
        now = datetime.datetime.utcnow()
        self.time_last_added = now
        delta = now - self.time_started
        delta_int = int(delta.total_seconds())
        num_tokens = sum(count_openai_token(entry.payload) for entry in responses)
        if delta_int not in self.trace:
            self.trace[delta_int] = num_tokens
        else:
            self.trace[delta_int] += num_tokens
        for entry in responses:
            self.responses.insert(
                entry.request_id, entry.offset, entry.payload, now, entry.finished
            )

        return True

    def count_total_openai_tokens(self) -> int:
        return sum(count_openai_token(entry.buffer) for entry in self.responses.content)

    def calculate_trace(self, time_range: int) -> list[int]:
        trace = []
        responses = list(
            map(
                lambda x: (count_openai_token(x.buffer), x.time_last_added),
                filter(lambda x: x.time_last_added is not None, self.responses.content),
            )
        )

        time_first_get = self.time_started

        for i in range(1, time_range + 1):
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

    def dump(self) -> None:
        with open(f"./tmp/{self.run_id}.pickle", "wb") as f:
            pickle.dump(self, f)


class LimitedTimeRun(BasicRun):
    """
    Currently, the timer goes when the struct is created.
    """

    def __init__(
        self,
        run_id: str,
        username: str,
        workload: Workload,
        time_limit: int,
    ):
        super().__init__(run_id, username, workload)
        self.sumuped = False
        self.time_limit = time_limit

    def check_limitation(self) -> bool:
        now = datetime.datetime.utcnow()
        return (not self.sumuped) and (
            now - self.time_started <= datetime.timedelta(seconds=self.time_limit)
        )

    def sumup(self) -> Summary:
        if not self.sumuped:
            self.dump()
            self.sumuped = True

        time_used = self.get_time_used()
        if time_used is None:
            time_used = float(self.time_limit)

        return Summary(
            run_id=self.run_id,
            username=self.username,
            time_used=time_used,
            num_token_total_llama3_avg=sum(
                count_llama3_token(entry.buffer) for entry in self.responses.content
            )
            / self.time_limit,
            num_token_total_openai_avg=self.count_total_openai_tokens()
            / self.time_limit,
            num_get_total=self.num_get_total,
            num_post_total=self.num_post_total,
            num_task_finished=self.num_post_total,
            num_task_touched=self.num_get_total,
            first_get_time=self.time_started,
            last_post_time=self.time_last_added,
            sample_requests_first_3=list(
                zip(self.workload.completions[:3], self.responses.content[:3])
            ),
            sample_requests_last_3=list(
                zip(self.workload.completions[-3:], self.responses.content[-3:])
            ),
            trace_num_token=self.calculate_trace(self.time_limit),
            better_trace=self.trace,
        )
