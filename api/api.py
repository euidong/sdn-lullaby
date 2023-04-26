from typing import List
from dataType import Server

# Python Interface. PEP 544
# https://peps.python.org/pep-0544/


class Api:
    srvs: List[Server]

    def reset(self) -> None:
        """set up system like getting initial state or generating new state"""

    def move_vnf(self, vnf_id: int, srv_id: int) -> bool:
        """Do move vnf from vnf_id to srv_id"""

    def get_util_from_srvs(self) -> List[Server]:
        """Do get util from srvs"""
