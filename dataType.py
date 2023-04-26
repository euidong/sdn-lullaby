from typing import List
from dataclasses import dataclass

"""
주어진 정보는 아래와 같아야 한다.

- Server
  - Server ID
  - Server CPU Core Capacity
  - Server Memory Capacity
  - Included VNFs

- VNF
  - VNF ID
  - VNF CPU Core Requirement
  - VNF Memory Requirement
  - SFC ID
"""

@dataclass
class VNF:
    id: int
    cpu_req: int
    mem_req: int
    sfc_id: int

@dataclass
class Server:
    id: int
    cpu_cap: int
    mem_cap: int
    vnfs: List[VNF]