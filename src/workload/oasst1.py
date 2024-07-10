from typing import Optional
from datasets import load_dataset

from .utils import cache
from .workload import ChatMessage, Workload, ChatCompletion


class OasstNode:
    def __init__(self, message: ChatMessage):
        self.message = message
        self.children: list[OasstNode] = []

    def add_child(self, node: "OasstNode"):
        self.children.append(node)

    def traverse_for_workload(self, messages: list[ChatMessage], workload: Workload):
        messages.append(self.message)
        completion = ChatCompletion(messages=messages)
        workload.completions.append(completion)
        for child in self.children:
            child.traverse_for_workload(messages.copy(), workload)


class OasstTree:
    def __init__(self):
        self.roots: list[OasstNode] = []
        self.id2node: dict[str, OasstNode] = {}

    def add_node(self, message_id: str, parent_id: Optional[str], text: str, role: str):
        message = ChatMessage(content=text, role=role)
        node = OasstNode(message)
        self.id2node[message_id] = node
        if parent_id is None:
            self.roots.append(node)
        else:
            parent = self.id2node[parent_id]
            parent.add_child(node)


class Oasst1Dataset:
    def __init__(self):
        self.raw_data = load_dataset("OpenAssistant/oasst1")

    @cache()
    def into_workload(self) -> Workload:
        merged_data = list(self.raw_data["train"]) + list(self.raw_data["validation"])  # type: ignore
        tree = OasstTree()
        for row in merged_data:
            tree.add_node(row["message_id"], row["parent_id"], row["text"], row["role"])  # type: ignore

        workload = Workload(completions=[])
        for root in tree.roots:
            root.traverse_for_workload([], workload)
        return workload


if __name__ == "__main__":
    workload = Oasst1Dataset().into_workload()
