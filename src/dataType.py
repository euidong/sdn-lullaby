from typing import List, Optional
from dataclasses import dataclass

import torch


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
    vnf_s_in: Optional[torch.Tensor]
    vnf_s_out: Optional[torch.Tensor]
    vnf_p_in: Optional[torch.Tensor]
    vnf_p_out: Optional[torch.Tensor]
    reward: Optional[torch.Tensor]
    next_vnf_s_in: Optional[torch.Tensor]
    next_vnf_p_in: Optional[torch.Tensor]
