from typing import List

from src.dataType import Edge, Server, VNF, SFC


class Api:
    edge: Edge
    srvs: List[Server]
    vnfs: List[VNF]
    sfcs: List[SFC]

    def reset(self) -> None:
        """set up system like getting initial state or generating new state"""

    def move_vnf(self, vnf_id: int, srv_id: int) -> bool:
        """Do move vnf from vnf_id to srv_id"""

    def get_srvs(self) -> List[Server]:
        """Do get util from srvs"""

    def get_vnfs(self) -> List[VNF]:
        """Do get util from vnfs"""

    def get_sfcs(self) -> List[SFC]:
        """Do get util from sfcs"""

    def get_edge(self) -> Edge:
        """Do get util from edge"""
