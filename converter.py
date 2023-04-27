from dataType import Server
from typing import Tuple, List


class Converter:
    def __init__(self, srv_n, vnf_n, cpu_cap, mem_cap):
        self.srv_n = srv_n
        self.vnf_n = vnf_n
        self.cpu_cap = cpu_cap
        self.mem_cap = mem_cap

    def encode_state(self, srvs: List[Server]) -> int:
        """Encode state to integer

        Args:
            srvs (List[Server]): List of servers

        Returns:
            int: encoded state
        """
        srv_rems = []
        for srv in srvs:
            srv_rems.append((srv.cpu_cap, srv.mem_cap))
            for vnf in srv.vnfs:
                srv_rems[srv.id][0] -= vnf.cpu_req
                srv_rems[srv.id][1] -= vnf.mem_req
        state = 0
        for i, (cpu_rem, mem_rem) in enumerate(srv_rems):
            state += ((self.cpu_cap * self.mem_cap) ** i) * \
                ((cpu_rem * self.mem_cap) + mem_rem)
        return state

    def decode_state(self, state: int) -> List[Server]:
        """Decode state to List of servers

        Args:
            state (int): encoded state
            srvs (List[Server]): List of servers

        Returns:
            List[Server]: List of servers
        """
        srv_rems = []
        for i in range(self.srv_n):
            rem = state // (self.cpu_cap * self.mem_cap)
            cpu_rem = rem // self.mem_cap
            mem_rem = rem % self.mem_cap
            srv_rems.append((cpu_rem, mem_rem))
            state -= rem
            state //= self.cpu_cap * self.mem_cap
        return srv_rems

    def encode_action(self, action: Tuple[int, int]) -> int:
        """Encode action to integer

        Args:
            action (Tuple[int, int]): (vnf_id, srv_id)
        Returns:
            int: encoded action
        """
        return self.srv_n * action[0] + action[1]

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

    def get_state_len(self) -> int:
        """Get state length

        Returns:
            int: state length
        """
        return (self.cpu_cap * self.mem_cap) ** self.srv_n

    def get_action_len(self) -> int:
        """Get action length

        Returns:
            int: action length
        """
        return self.srv_n * self.vnf_n
