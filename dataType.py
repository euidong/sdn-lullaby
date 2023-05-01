from typing import List, Optional
from dataclasses import dataclass
import torch

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
    srv_id: int

@dataclass
class Server:
    id: int
    cpu_cap: int
    mem_cap: int
    cpu_load: int
    mem_load: int
    vnfs: List[VNF]

@dataclass
class Edge:
    cpu_cap: int
    mem_cap: int
    cpu_load: int
    mem_load: int

@dataclass
class SFC:
    id: int
    vnfs: List[VNF]

@dataclass
class State:
    edge: Edge
    srvs: List[Server]
    vnfs: List[VNF]
    sfcs: List[SFC]

@dataclass
class Action:
    vnf_id: int
    srv_id: int

@dataclass
class Scene:
    vm_s_in: Optional[torch.Tensor]
    vm_s_out: Optional[torch.Tensor]
    vm_p_in: Optional[torch.Tensor]
    vm_p_out: Optional[torch.Tensor]
    reward: Optional[torch.Tensor]
    next_vm_s_in: Optional[torch.Tensor]
    next_vm_p_in: Optional[torch.Tensor]