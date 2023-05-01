from api.api import Api
from typing import List
from dataType import Edge, Server, VNF, SFC


class NiTestBed(Api):
    edge: Edge
    srvs: List[Server]
    vnfs: List[VNF]
    sfcs: List[SFC]

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def move_vnf(self, vnf_id: int, srv_id: int) -> bool:
        pass

    def get_srvs(self) -> List[Server]:
        pass

    def get_vnfs(self) -> List[VNF]:
        pass

    def get_sfcs(self) -> List[SFC]:
        pass

    def get_edge(self) -> Edge:
        pass
