from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
import datetime
import uuid
import jwt
from openai import BaseModel

from .workload.workload import ChatCompletion
from .workload.oasst1 import Oasst1Dataset
from .run import ResponseChunk, Run, Summary

app = FastAPI()
key = "secret"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
users: dict[str, str] = {}
user2run: dict[str, set[str]] = {}
runs: dict[str, Run] = {}


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
    user2run[form.username] = set()
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
    run_id = str(uuid.uuid4())
    user2run[user].add(run_id)
    runs[run_id] = Run(
        username=user,
        rule=request.rule,
        workload=Oasst1Dataset().into_workload(),
    )
    return {"run_id": run_id}


@app.get("/run/lt/{run_id}/batch", response_model=list[ChatCompletion])
def get_batch(
    run_id: str, offset: int, limit: int, sorted: bool, user=Depends(authenticate)
):
    if run_id not in runs or run_id not in user2run[user]:
        raise HTTPException(status_code=404, detail="Run not found")

    run = runs[run_id]
    completions = run.get_workload(offset, limit)
    if completions is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Time limit exceeded"
        )
    return completions


@app.post("/run/lt/{run_id}")
def submit_results(
    run_id: str,
    chunk: ResponseChunk,
    user=Depends(authenticate),
):
    if run_id not in runs or run_id not in user2run[user]:
        raise HTTPException(status_code=404, detail="Run not found")

    run = runs[run_id]
    if not run.add_result(chunk):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Time limit exceeded"
        )
    return {"message": "ok"}


@app.get("/run/lt/{run_id}/summary", response_model=Summary)
def get_summary(run_id: str, user=Depends(authenticate)):
    if run_id not in runs or run_id not in user2run[user]:
        raise HTTPException(status_code=404, detail="Run not found")

    run = runs[run_id]
    return run.sumup()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=28000)
