import os
from typing import List

from src.env import Environment
from src.utils import save_animation
from src.api.simulator import Simulator
from src.dataType import State, Action

class BaselineRuleBasedAgent:
    def decide_action(self, state: State) -> Action:
        sorted_srv_idxs = self._get_sorted_srv_idxs_with_srv_load(state)
        while len(sorted_srv_idxs) > 0:
            src_srv_id = self._select_srv(sorted_srv_idxs)
            sorted_vnf_idxs = self._get_sorted_vnf_idxs_with_vnf_req(state, src_srv_id)
            while len(sorted_vnf_idxs) > 0:
                vnf_id = self._select_vnf(sorted_vnf_idxs)
                possible_tgt_srv_idxs = self._get_possible_tgt_srv_idxs_with_srv_load(state, src_srv_id, vnf_id)
                tgt_srv_id = self._place_vnf(possible_tgt_srv_idxs)
                if tgt_srv_id is not None:
                    return Action(vnf_id, tgt_srv_id)
                vnf_id = self._select_vnf(sorted_vnf_idxs)
        return Action(-1, -1)

    def _get_sorted_srv_idxs_with_srv_load(self, state: State) -> List[int]:
        srv_loads = []
        for srv in state.srvs:
            srv_load = srv.cpu_load / srv.cpu_cap + srv.mem_load / srv.mem_cap
            srv_loads.append((srv.id, srv_load))
        sorted_srv_loads = sorted(srv_loads, key=lambda x: x[1])
        sorted_srv_idxs = [x[0] for x in sorted_srv_loads]
        return sorted_srv_idxs

    def _get_sorted_vnf_idxs_with_vnf_req(self, state: State, src_srv_id: int) -> List[int]:
        for srv in state.srvs:
            if srv.id == src_srv_id:
                src_srv = srv
                break
        vnf_reqs = []
        for vnf in src_srv.vnfs:
            vnf_req = vnf.cpu_req / src_srv.cpu_cap + vnf.mem_req / src_srv.mem_cap
            vnf_reqs.append((vnf.id, vnf_req))
        sorted_vnf_reqs = sorted(vnf_reqs, key=lambda x: x[1])
        sorted_vnf_idxs = [x[0] for x in sorted_vnf_reqs]
        return sorted_vnf_idxs

    def _get_possible_tgt_srv_idxs_with_srv_load(self, state: State, src_srv_id: int, vnf_id: int) -> List[int]:
        for srv in state.srvs:
            if srv.id == src_srv_id:
                src_srv = srv
                break
        for v in src_srv.vnfs:
            if v.id == vnf_id:
                vnf = v
                break
        possible_tgt_srv_idxs = []
        for tgt_srv in state.srvs:
            tgt_srv_id = tgt_srv.id
            if tgt_srv_id == src_srv_id:
                continue
            if tgt_srv.cpu_load + vnf.cpu_req > tgt_srv.cpu_cap:
                continue
            if tgt_srv.mem_load + vnf.mem_req > tgt_srv.mem_cap:
                continue
            possible_tgt_srv_idxs.append(tgt_srv_id)
        return possible_tgt_srv_idxs

    def _select_srv(self, sorted_srv_idxs: List[int]) -> int:
        min_load_srv_idx = sorted_srv_idxs.pop(0)
        return min_load_srv_idx
    
    def _select_vnf(self, sorted_vnf_idxs: List[int]) -> int:
        min_req_vnf_idx = sorted_vnf_idxs.pop(0)
        return min_req_vnf_idx
    
    def _place_vnf(self, possible_tgt_srv_idxs: List[int]) -> int:
        if len(possible_tgt_srv_idxs) == 0:
            return None
        tgt_srv_id = possible_tgt_srv_idxs.pop(-1)
        return tgt_srv_id


def evaluate(agent: BaselineRuleBasedAgent, make_env_fn, seed, file_name):
    env = make_env_fn(seed)
    state = env.reset()
    history = []
    while True:
        action = agent.decide_action(state)
        state, reward, done = env.step(action)
        history.append((state, action))
        if done:
            break
    history.append((state, None))
    os.makedirs('result/baseline-rule', exist_ok=True)
    save_animation(
        srv_n=srv_n, sfc_n=sfc_n, vnf_n=max_vnf_num,
        srv_mem_cap=srv_mem_cap, srv_cpu_cap=srv_cpu_cap, 
        history=history, path=f'./result/baseline-rule/{file_name}.mp4',
    )

if __name__ == '__main__':
    # Simulator Args
    srv_n = 8
    sfc_n = 8
    max_vnf_num = 30
    srv_cpu_cap = 16
    srv_mem_cap = 64
    max_edge_load = 0.6
    seed=927
    
    make_env_fn = lambda seed : Environment(
        api=Simulator(srv_n, srv_cpu_cap, srv_mem_cap, max_vnf_num, sfc_n, max_edge_load),
        seed=seed,
    )

    agent = BaselineRuleBasedAgent()

    evaluate(agent, make_env_fn, seed, file_name='init')
