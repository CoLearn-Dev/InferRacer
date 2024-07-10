from pydantic import BaseModel


class ChatMessage(BaseModel):
    content: str
    role: str


class ChatCompletion(BaseModel):
    messages: list[ChatMessage]


class Workload(BaseModel):
    completions: list[ChatCompletion]
