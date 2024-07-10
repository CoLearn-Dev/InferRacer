from workload.workload import Workload


class Competition:
    def __init__(self, workload: Workload, time_limit: int = 60):
        self.workload = workload
        self.time_limit = time_limit
