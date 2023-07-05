from collections import deque
from typing import List, Optional

import itertools
import numpy as np


class ReplayMemory:
    def __init__(self, batch_size, max_memory_len=1_000):
        self.batch_size = batch_size
        self.max_memory_len = max_memory_len
        self.buffer = deque(maxlen=max_memory_len)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self) -> List[any]:
        if len(self.buffer) < self.batch_size:
            return []
        return np.random.choice(list(itertools.islice(
            self.buffer, 0, len(self.buffer) - 1)), self.batch_size)

    def append(self, data: any) -> None:
        self.buffer.append(data)

    def last(self, n: int) -> List[any]:
        return list(itertools.islice(self.buffer, max(0, len(self.buffer)-n), len(self.buffer)))
