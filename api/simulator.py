from typing import List, Tuple
from dataType import Edge, Server, VNF, SFC
from api.api import Api
import numpy as np

class Simulator(Api):
    edge: Edge
    srvs: List[Server]
    vnfs: List[VNF]
    sfcs: List[SFC]

    def __init__(self, srv_n: int = 4, srv_cpu_cap: int = 8, srv_mem_cap: int = 32, max_vnf_num: int = 100) -> None:
        """Intialize Simulator

        Args:
            srv_n (int): server number
            srv_cpu_cap (int): each server's capcaity of cpu
            srv_mem_cap (int): each server's capacity of memory
        """
        self.srv_n = srv_n
        self.srv_cpu_cap = srv_cpu_cap
        self.srv_mem_cap = srv_mem_cap
        self.max_vnf_num = max_vnf_num
        
        self.edge = Edge(
            cpu_cap=srv_cpu_cap * srv_n,
            mem_cap=srv_mem_cap * srv_n,
            cpu_load=0,
            mem_load=0
        )
        self.srvs = []
        self.vnfs = []
        self.sfcs = []
        for i in range(srv_n):
            self.srvs.append(Server(
                id=i, 
                cpu_cap=srv_cpu_cap, 
                mem_cap=srv_mem_cap,
                cpu_load=0,
                mem_load=0,
                vnfs=[],
            ))
        
    # TODO: need a logic for mixing vnf.
    def reset(self, sfc_n: int = 4, max_edge_util: float = 0.1) -> None:
        """Generate random VNFs and put them into servers

        Args:
            sfc_n (int, optional): number of SFCs. Defaults to 4.
            max_edge_load (float, optional): maximum edge load. Defaults to 0.3.
        """

        # 초기화
        self.edge.cpu_load = 0
        self.edge.mem_load = 0
        self.srvs = []
        self.vnfs = []
        self.sfcs = []
        for i in range(self.srv_n):
            self.srvs.append(Server(
                id=i, 
                cpu_cap=self.srv_cpu_cap, 
                mem_cap=self.srv_mem_cap,
                cpu_load=0,
                mem_load=0,
                vnfs=[],
            ))

        # 최소한 하나의 VNF(CPU=1, Mem=1)을 가진 SFC를 생성
        # 최대한 각 서버에 골고루 분배
        sfcs = [SFC(id=i, vnfs=[]) for i in range(sfc_n)]
        vnf_cnt = 0
        for i in range(sfc_n):
            srv_id = i % len(self.srvs)
            vnf = VNF(
                id=i, 
                cpu_req=1,
                mem_req=1,
                sfc_id=i,
                srv_id=srv_id
            )
            self.vnfs.append(vnf)
            sfcs[i].vnfs.append(vnf)
            vnf_cnt += 1
            self.srvs[srv_id].vnfs.append(vnf)
            self.srvs[srv_id].cpu_load += vnf.cpu_req
            self.srvs[srv_id].mem_load += vnf.mem_req
            self.edge.cpu_load += vnf.cpu_req
            self.edge.mem_load += vnf.mem_req

        # Possible VNF Type (CPU req, Memr req)
        POSSIBLE_VNF_TYPE = [
            (1, 1), (1, 2), (1, 4),
            (2, 1), (2, 2), (2, 4), (2, 8),
            (4, 4), (4, 8), (4, 16),
            (8, 4), (8, 8), (8, 16), (8, 32)
        ]

        
        while self.edge.cpu_load / self.edge.cpu_cap  < max_edge_util and self.edge.mem_load / self.edge.mem_cap < max_edge_util and vnf_cnt < self.max_vnf_num:
            # VNF를 생성
            vnf_type = POSSIBLE_VNF_TYPE[np.random.choice(len(POSSIBLE_VNF_TYPE))]
            vnf = VNF(id=vnf_cnt, 
                      cpu_req=vnf_type[0],
                      mem_req=vnf_type[1],
                      sfc_id=np.random.randint(sfc_n),
                      srv_id=-1
                      )
            
            # 저장할 서버 선택
            srv_id = np.random.randint(len(self.srvs))

            # 저장 가능한지 확인
            srv_remain_cpu_cap = self.srvs[srv_id].cpu_cap - self.srvs[srv_id].cpu_load
            srv_remain_mem_cap = self.srvs[srv_id].mem_cap - self.srvs[srv_id].mem_load
            if srv_remain_cpu_cap < vnf.cpu_req or srv_remain_mem_cap < vnf.mem_req:
                continue

            # VNF를 서버에 할당
            vnf.srv_id = srv_id
            self.srvs[srv_id].vnfs.append(vnf)
            self.srvs[srv_id].cpu_load += vnf.cpu_req
            self.srvs[srv_id].mem_load += vnf.mem_req

            # sfcs에 추가
            sfcs[vnf.sfc_id].vnfs.append(vnf)

            # vnfs에 추가
            self.vnfs.append(vnf)

            self.edge.cpu_load += vnf.cpu_req
            self.edge.mem_load += vnf.mem_req

            vnf_cnt += 1
        self.sfcs = sfcs

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
        srv_remain_cpu_cap = self.srvs[srv_id].cpu_cap - self.srvs[srv_id].cpu_load
        srv_remain_mem_cap = self.srvs[srv_id].mem_cap - self.srvs[srv_id].mem_load
        if srv_remain_cpu_cap < target_vnf.cpu_req or srv_remain_mem_cap <target_vnf.mem_req:
            return False
        # vnf 검색 및 이동 (없으면 False 리턴)
        for srv in self.srvs:
            for vnf in srv.vnfs:
                if vnf.id == vnf_id:
                    vnf.srv_id = srv_id
                    self.srvs[srv_id].vnfs.append(vnf)
                    self.srvs[srv_id].cpu_load += vnf.cpu_req
                    self.srvs[srv_id].mem_load += vnf.mem_req
                    
                    srv.vnfs.remove(vnf)
                    srv.cpu_load -= vnf.cpu_req
                    srv.mem_load -= vnf.mem_req
                    return True
        return False

    def get_srvs(self) -> List[Server]:
        return self.srvs

    def get_vnfs(self) -> List[VNF]:
        return self.vnfs

    def get_sfcs(self) -> List[List[VNF]]:
        return self.sfcs

    def get_edge(self) -> Edge:
        return self.edge