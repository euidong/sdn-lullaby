from api.api import Api
from typing import List
from dataType import Server


class NiTestBed(Api):
    srvs: List[Server]

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def move_vnf(self, vnf_id: int, srv_id: int) -> bool:
        pass

    def get_util_from_srvs(self) -> List[Server]:
        pass
