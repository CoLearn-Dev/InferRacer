from typing import Optional
from openai import BaseModel
import uuid

from .run import LimitedTimeRun
from .workload.workload import Workload


class RankingSlot(BaseModel):
    run_id: str
    num_token_total_openai_avg: float
    rank: int
    username: str


class LimitedTimeCompetition:
    def __init__(self, workload: Workload, time_limit: int = 60):
        self.workload = workload
        self.time_limit = time_limit
        self.user2run: dict[str, set[str]] = {}
        self.runs: dict[str, LimitedTimeRun] = {}

    def create_run(
        self, username: str, time_limit: Optional[int], shuffled: bool
    ) -> LimitedTimeRun:
        if time_limit is None:
            time_limit = self.time_limit
        if shuffled:
            run_id = f"lt-{time_limit}-shuffled-{uuid.uuid4()}"
        else:
            run_id = f"lt-{time_limit}-noshuffled-{uuid.uuid4()}"
        run = LimitedTimeRun(run_id, username, self.workload, time_limit)
        if username not in self.user2run:
            self.user2run[username] = set()
        self.user2run[username].add(run_id)
        self.runs[run_id] = run
        return run

    def get_leaderboard(self) -> list[RankingSlot]:
        results = []
        for username, run_ids in self.user2run.items():
            for run_id in run_ids:
                run = self.runs[run_id]
                tokens = run.count_total_openai_tokens()
                time_used = run.get_time_used()
                if time_used is None:
                    time_used = float(run.time_limit)
                results.append(
                    RankingSlot(
                        run_id=run_id,
                        num_token_total_openai_avg=tokens / time_used,
                        rank=0,
                        username=username,
                    )
                )

        results.sort(key=lambda x: x.num_token_total_openai_avg, reverse=True)
        for i, result in enumerate(results):
            if i == 0:
                result.rank = 1
            else:
                if (
                    result.num_token_total_openai_avg
                    == results[i - 1].num_token_total_openai_avg
                ):
                    result.rank = results[i - 1].rank
                else:
                    result.rank = i + 1

        return results
