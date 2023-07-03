import os

import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass

from dataType import State
from const import VNF_SELECTION_IN_DIM, VNF_PLACEMENT_IN_DIM

@dataclass
class DebugInfo:
    timestamp: str
    episode: int
    step: int
    mean_100_change_slp_srv: float
    mean_100_init_slp_srv: float
    mean_100_final_slp_srv: float
    srv_n: int
    mean_100_change_sfc_in_same_srv: float
    mean_100_init_sfc_in_same_srv: float
    mean_100_final_sfc_in_same_srv: float
    sfc_n: int
    mean_100_exploration: float

def setup_mp_env():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'

def print_debug_info(debug_info: DebugInfo, refresh: bool = False):
    debug_msg = "[{}] Episode {:05}, Step {:04.2f}, #SleepSrv ({:02.3f})({:02.3f}->{:02.3f}/{}), #SFCinSameSrv ({:02.3f})({:02.3f}->{:02.3f}/{}), #Exploration: {:.3f}".format(
        debug_info.timestamp, debug_info.episode, debug_info.step,
        debug_info.mean_100_change_slp_srv, debug_info.mean_100_init_slp_srv, debug_info.mean_100_final_slp_srv, debug_info.srv_n,
        debug_info.mean_100_change_sfc_in_same_srv, debug_info.mean_100_init_sfc_in_same_srv, debug_info.mean_100_final_sfc_in_same_srv, debug_info.sfc_n,
        debug_info.mean_100_exploration,
    )
    print(debug_msg, end='\r', flush=True)
    if refresh:
        print('\x1b[2K' + debug_msg, flush=True)

def convert_state_to_vnf_selection_input(state: State, max_vnf_num: int) -> torch.Tensor:
    vnf_selection_input = torch.zeros(max_vnf_num, VNF_SELECTION_IN_DIM, dtype=torch.float32)
    for vnf in state.vnfs:
        vnf_selection_input[vnf.id] = torch.tensor([
            vnf.cpu_req, vnf.mem_req, vnf.sfc_id,
            state.srvs[vnf.srv_id].cpu_cap, state.srvs[vnf.srv_id].mem_cap,
            state.srvs[vnf.srv_id].cpu_load, state.srvs[vnf.srv_id].mem_load,
            state.edge.cpu_cap, state.edge.mem_cap,
            state.edge.cpu_load, state.edge.mem_load,
        ])
    return vnf_selection_input

def convert_state_to_vnf_placement_input(state: State, vnf_id: int) -> torch.Tensor:
    vnf_placement_input = torch.zeros(len(state.srvs), VNF_PLACEMENT_IN_DIM, dtype=torch.float32)
    for srv in state.srvs:
        vnf_placement_input[srv.id] = torch.tensor([
            srv.cpu_cap, srv.mem_cap, srv.cpu_load, srv.mem_load,
            state.vnfs[vnf_id].cpu_req, state.vnfs[vnf_id].mem_req, state.vnfs[vnf_id].sfc_id,
            state.edge.cpu_cap, state.edge.mem_cap, state.edge.cpu_load, state.edge.mem_load
        ])
    return vnf_placement_input

def get_possible_actions(state: State, max_vnf_num: int) -> Dict[int, List[int]]:
    '''return possible actions for each state

    Args:
        state (State): state

    Returns:
        Dict[int, List[int]]: possible actions
                                    ex) {vnfId: [srvId1, srvId2, ...], vnfId2: [srvId1, srvId2, ...], ...}
    '''
    possible_actions = {}
    for vnf_idx in range(max_vnf_num):
        possible_actions[vnf_idx] = []
        if len(state.vnfs) <= vnf_idx: continue
        vnf = state.vnfs[vnf_idx]
        for srv in state.srvs:
            # 동일한 srv로 다시 전송하는 것 방지
            if vnf.srv_id == srv.id: continue
            # capacity 확인
            if srv.cpu_cap - srv.cpu_load < vnf.cpu_req or srv.mem_cap - srv.mem_load < vnf.mem_req: continue
            possible_actions[vnf.id].append(srv.id)
    return possible_actions

def get_info_from_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = logit_to_prob(logits)
    dist = torch.distributions.Categorical(probs=probs)
    actions = dist.sample().to(torch.int32)
    logpas = dist.log_prob(actions)
    is_exploratory = actions != torch.argmax(logits, dim=1)
    return actions, logpas, is_exploratory

def logit_to_prob(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.zeros_like(logits)
    # 0인 값은 prob을 0으로 유지하고, 나머지 값을 확률로 변경
    for i in range(logits.shape[0]):
        probs[i, logits[i] != 0] = torch.softmax(logits[i, logits[i] != 0], dim = 0)
    return probs

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_zero_util_cnt(state: State) -> int:
    cnt = 0
    for srv in state.srvs:
        if len(srv.vnfs) == 0:
            cnt += 1
    return cnt
    
def get_sfc_cnt_in_same_srv(state: State) -> int:
    cnt = 0
    for sfc in state.sfcs:
        if len(sfc.vnfs) == 0:
            continue
        cnt += 1
        srv_id = sfc.vnfs[0].srv_id
        for vnf in sfc.vnfs:
            if srv_id != vnf.srv_id:
                cnt -= 1
                break
    return cnt