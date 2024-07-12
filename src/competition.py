from typing import Optional
from openai import BaseModel
from .run import LimitedTimeRun
from .workload.workload import Workload


class RankingSlot(BaseModel):
    run_id: str
    num_token_total_openai: int
    rank: int
    username: str


class LimitedTimeCompetition:
    def __init__(self, workload: Workload, time_limit: int = 60):
        self.workload = workload
        self.time_limit = time_limit
        self.user2run: dict[str, set[str]] = {}
        self.runs: dict[str, LimitedTimeRun] = {}

    def create_run(self, username: str, run_id: str, rule: str) -> LimitedTimeRun:
        run = LimitedTimeRun(username, rule, self.workload)
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
                results.append(
                    RankingSlot(
                        run_id=run_id,
                        num_token_total_openai=tokens,
                        rank=0,
                        username=username,
                    )
                )

        results.sort(key=lambda x: x.num_token_total_openai, reverse=True)
        for i, result in enumerate(results):
            if i == 0:
                result.rank = 1
            else:
                if (
                    result.num_token_total_openai
                    == results[i - 1].num_token_total_openai
                ):
                    result.rank = results[i - 1].rank
                else:
                    result.rank = i + 1

        return results
