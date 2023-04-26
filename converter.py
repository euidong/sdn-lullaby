from dataType import Server
from typing import Tuple


class Converter:
    def __init__(self, srv_n, max_vnf_num):
        self.srv_n = srv_n
        self.max_vnf_num = max_vnf_num

    def encode_state(self, srvs: list[Server]) -> int:
        """Encode state to integer

        Args:
            srvs (list[Server]): list of servers

        Returns:
            int: encoded state
        """
        state = 0
        for srv in srvs:
            state += (srv.cpu_cap - srv.cpu_remain) * srv.mem_cap
        return state

    def decode_state(self, state: int, srvs: list[Server]) -> list[Server]:
        """Decode state to list of servers

        Args:
            state (int): encoded state
            srvs (list[Server]): list of servers

        Returns:
            list[Server]: list of servers
        """
        for srv in srvs:
            srv.cpu_remain = srv.cpu_cap - state % srv.mem_cap
            state //= srv.mem_cap
        return srvs

    @classmethod
    def encode_action(self, action: Tuple[int, int]) -> int:
        """Encode action to integer

        Args:
            action (Tuple[int, int]): (vnf_id, srv_id)
        Returns:
            int: encoded action
        """
        return self.srv_n * action[0] + action[1]

    @classmethod
    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action to integer

        Args:
            action (int): encoded action

        Returns:
            Tuple[int, int]: (vnf_id, srv_id)
        """
        vnf_id = action // self.srv_n
        srv_id = action % self.srv_n
        return (vnf_id, srv_id)
