from datasets import load_dataset

from .utils import cache
from .workload import Workload, ChatCompletion


class ArenaDataset:
    def __init__(self):
        self.raw_data = load_dataset("lmsys/chatbot_arena_conversations")

    @cache()
    def into_workload(self) -> Workload:
        train_data = self.raw_data["train"]  # type: ignore
        conversation_a = train_data["conversation_a"]  # type: ignore
        workload = Workload(completions=[])
        for conversation in conversation_a:
            for i in range(0, len(conversation), 2):
                completion = ChatCompletion(messages=conversation[: i + 1])
                workload.completions.append(completion)
        return workload


if __name__ == "__main__":
    workload = ArenaDataset().into_workload()
