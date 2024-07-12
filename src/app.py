from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
import datetime
import uuid
import jwt
import json
from openai import BaseModel

from .competition import LimitedTimeCompetition, RankingSlot

from .workload.oasst1 import Oasst1Dataset
from .run import RequestEntry, ResponseEntry, LimitedTimeRun, Summary

app = FastAPI()
key = "secret"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
users: dict[str, str] = {}
competition = LimitedTimeCompetition(Oasst1Dataset().into_workload())


class CreateRunRequest(BaseModel):
    rule: str
    model: str


class SubmitResultsRequest(BaseModel):
    offset: int
    chunk_id: int
    finished: bool
    data: list[str]


@app.post("/user/signin")
def signin(form: OAuth2PasswordRequestForm = Depends()):
    if form.username in users:
        raise HTTPException(status_code=400, detail="User already exists")
    users[form.username] = form.password
    return {"username": form.username}


@app.post("/user/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username not in users:
        raise HTTPException(status_code=400, detail="User not found")
    if users[form.username] != form.password:
        raise HTTPException(status_code=400, detail="Invalid password")
    token = jwt.encode(
        {
            "username": form.username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        },
        key,
        algorithm="HS256",
    )
    return {"token": token}


def authenticate(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, key, algorithms=["HS256"])
        username: Optional[str] = payload.get("username")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    return username


@app.post("/run/lt")
async def create_run(request: CreateRunRequest, user=Depends(authenticate)):
    run = competition.create_run(user, request.rule)
    return {"run_id": run.run_id}


@app.get("/run/lt/{run_id}/fetch", response_model=list[RequestEntry])
def get_batch(
    run_id: str, offset: int, limit: int, sorted: bool, user=Depends(authenticate)
):
    if run_id not in competition.runs or run_id not in competition.user2run[user]:
        raise HTTPException(status_code=404, detail="Run not found")

    run = competition.runs[run_id]
    completions = run.get_workload(offset, limit)
    if completions is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Time limit exceeded"
        )
    return completions


async def process_responses(run: LimitedTimeRun, request: Request):
    async for bs in request.stream():
        if bs == b"":
            break
        for raw in bs.decode("utf-8").split("\n"):
            if raw == "":
                continue
            data = json.loads(raw)
            entries = [ResponseEntry(**item) for item in data]
            if not run.add_result(entries):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Time limit exceeded",
                )


@app.post("/run/lt/{run_id}/push")
async def push_results(
    run_id: str,
    request: Request,
    user=Depends(authenticate),
):
    if run_id not in competition.runs or run_id not in competition.user2run[user]:
        raise HTTPException(status_code=404, detail="Run not found")

    run = competition.runs[run_id]
    await process_responses(run, request)
    return {"message": "ok"}


@app.post("/run/lt/{run_id}/submit")
def submit_results(
    run_id: str,
    entries: list[ResponseEntry],
    user=Depends(authenticate),
):
    if run_id not in competition.runs or run_id not in competition.user2run[user]:
        raise HTTPException(status_code=404, detail="Run not found")

    run = competition.runs[run_id]
    if not run.add_result(entries):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Time limit exceeded"
        )
    return {"message": "ok"}


@app.get("/run/lt/{run_id}/summary", response_model=Summary)
def get_summary(run_id: str, user=Depends(authenticate)):
    if run_id not in competition.runs or run_id not in competition.user2run[user]:
        raise HTTPException(status_code=404, detail="Run not found")

    run = competition.runs[run_id]
    return run.sumup()


@app.get("/run/lt/leaderboard", response_model=list[RankingSlot])
def get_leaderboard(_=Depends(authenticate)):
    return competition.get_leaderboard()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=28000)
