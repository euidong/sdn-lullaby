from typing import List
from dataType import Server, VNF
from api.api import Api
import numpy as np

class Simulator(Api):
    srvs: List[Server]

    def __init__(self, srv_n: int, srv_cpu_cap: int, srv_mem_cap: int) -> None:
        """Intialize Simulator

        Args:
            srv_n (int): server number
            srv_cpu_cap (int): each server's capcaity of cpu
            srv_mem_cap (int): each server's capacity of memory
        """
        self.srvs = []
        for i in range(srv_n):
            self.srvs.append(Server(i, srv_cpu_cap, srv_mem_cap, []))

    def reset(self, sfc_n: int = 4, max_edge_load: float = 0.5) -> None:
        """Generate random VNFs and put them into servers

        Args:
            sfc_n (int, optional): number of SFCs. Defaults to 4.
            max_edge_load (float, optional): maximum edge load. Defaults to 0.5.
        """

        # 최소한 하나의 VNF(CPU=1, Mem=1)을 가진 SFC를 생성
        # 최대한 각 서버에 골고루 분배
        sfcs = [[] for _ in range(sfc_n)]
        vnf_cnt = 0
        for i in range(sfc_n):
            sfcs[i].append(VNF(i, 1, 1, i))
            vnf_cnt += 1
            self.srvs[i % len(self.srvs)].vnfs.append(sfcs[i][0])

        # Possible VNF Type (CPU req, Memr req)
        POSSIBLE_VNF_TYPE = [
            (1, 1), (1, 2), (1, 4),
            (2, 1), (2, 2), (2, 4), (2, 8),
            (4, 4), (4, 8), (4, 16),
            (8, 4), (8, 8), (8, 16), (8, 32)
        ]

        cpu_load, mem_load = self._calc_edge_load()
        while cpu_load < max_edge_load and mem_load < max_edge_load:
            # VNF를 생성
            vnf_type = POSSIBLE_VNF_TYPE[np.random.choice(len(POSSIBLE_VNF_TYPE))]
            vnf = VNF(vnf_cnt, vnf_type[0],
                      vnf_type[1], np.random.randint(sfc_n))
            

            # 저장할 서버 선택
            srv_id = np.random.randint(len(self.srvs))

            # 저장 가능한지 확인
            srv_remain_cpu_cap, srv_remain_mem_cap = self._calc_remain_cap_from_srv(
                srv_id)
            if srv_remain_cpu_cap < vnf.cpu_req or srv_remain_mem_cap < vnf.mem_req:
                continue

            # VNF를 서버에 할당
            self.srvs[srv_id].vnfs.append(vnf)

            # sfcs 추가
            sfcs[vnf.sfc_id].append(vnf)

            cpu_load, mem_load = self._calc_edge_load()
            vnf_cnt += 1

    def move_vnf(self, vnf_id: int, srv_id: int) -> bool:
        # vnf_id가 존재하는지 확인
        target_vnf = None
        for srv in self.srvs:
            for vnf in srv.vnfs:
                if vnf.id == vnf_id:
                    target_vnf = vnf
                    break
            if target_vnf is not None:
                break
        if target_vnf is None:
            return False
        # srv_id가 존재하는지 확인
        if srv_id >= len(self.srvs):
            return False
        # 해당 srv에 이미 vnf가 존재하는지 확인
        for vnf in self.srvs[srv_id].vnfs:
            if vnf.id == vnf_id:
                return False
        # capacity 확인
        srv_remain_cpu_cap, srv_remain_mem_cap = self._calc_remain_cap_from_srv(srv_id)
        if srv_remain_cpu_cap < target_vnf.cpu_req or srv_remain_mem_cap <target_vnf.mem_req:
            return False
        # vnf 검색 및 이동 (없으면 False 리턴)
        for srv in self.srvs:
            for vnf in srv.vnfs:
                if vnf.id == vnf_id:
                    self.srvs[srv_id].vnfs.append(vnf)
                    srv.vnfs.remove(vnf)
                    return True
        return False

    def get_util_from_srvs(self) -> List[Server]:
        return self.srvs

    def _calc_edge_load(self) -> List[float]:
        edge_cpu_cap = 0
        edge_mem_cap = 0
        edge_cpu_req = 0
        edge_mem_req = 0
        for srv in self.srvs:
            edge_cpu_cap += srv.cpu_cap
            edge_mem_cap += srv.mem_cap
            for vnf in srv.vnfs:
                edge_cpu_req += vnf.cpu_req
                edge_mem_req += vnf.mem_req
        return (edge_cpu_req / edge_cpu_cap, edge_mem_req / edge_mem_cap)

    def _calc_remain_cap_from_srv(self, srv_id) -> List[float]:
        srv_remain_cpu_cap = self.srvs[srv_id].cpu_cap
        srv_remain_mem_cap = self.srvs[srv_id].mem_cap
        for vnf in self.srvs[srv_id].vnfs:
            srv_remain_cpu_cap -= vnf.cpu_req
            srv_remain_mem_cap -= vnf.mem_req
        return [srv_remain_cpu_cap, srv_remain_mem_cap]
