from collections import deque
import numpy as np
import itertools
from typing import List, Optional, Tuple


class Memory:
    def __init__(self, batch_size, max_memory_len=100_000):
        self.batch_size = batch_size
        self.max_memory_len = max_memory_len
        self.memory = {}

    def __len__(self) -> int:
        return len(self.memory.keys())

    def sample(self, vnf_no: Optional[int] = None) -> List[any]:
        if not vnf_no:
            vnf_no = np.random.choice(list(self.memory.keys()))
        if vnf_no in self.memory:
            return []
        if len(self.memory[vnf_no]) < self.batch_size:
            return []
        return np.random.choice(list(itertools.islice(
            self.memory[vnf_no], 0, len(self.memory[vnf_no]) - 1)), self.batch_size)

    def append(self, vnf_no: int, data: any) -> None:
        if not vnf_no in self.memory:
            self.memory[vnf_no] = deque(maxlen=self.max_memory_len)
        self.memory[vnf_no].append(data)

    def last(self, vnf_no: int, n: int) -> List[any]:
        return list(itertools.islice(self.memory[vnf_no], max(0, len(self.memory)-n), len(self.memory)))
