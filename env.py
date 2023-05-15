from typing import List, Tuple
from api.api import Api
from dataType import Edge, Server, VNF, SFC, State, Action
from copy import deepcopy


class Environment:
    def __init__(self, api: Api) -> None:
        self.api = api

    # return next_state, reward, done
    def step(self, action: Action) -> Tuple[State, int, bool]:
        state = self._get_state()
        is_moved = self._move_vnf(action.vnf_id, action.srv_id)
        next_state = self._get_state()
        reward = self._calc_reward(is_moved, state, next_state)
        done = False
        return (next_state, reward, done)

    def reset(self) -> State:
        self.api.reset()
        return self._get_state()

    def _move_vnf(self, vnfId, srvId) -> None:
        return self.api.move_vnf(vnfId, srvId)

    def _get_srvs(self) -> List[Server]:
        return self.api.get_srvs()
    
    def _get_vnfs(self) -> List[VNF]:
        return self.api.get_vnfs()

    def _get_sfcs(self) -> List[SFC]:
        return self.api.get_sfcs()

    def _get_edge(self) -> Edge:
        return self.api.get_edge()

    def _calc_reward(self, is_moved: bool, state: State, next_state: State) -> int:
        # reward = 0
        # # 만약, 특정 server의 전원이 꺼졌다면, reward를 더 준다.
        # # 반대라면, reward를 감소 시킨다.
        # zero_util_cnt = self._get_zero_util_cnt(state)
        # next_zero_util_cnt = self._get_zero_util_cnt(next_state)
        # if zero_util_cnt < next_zero_util_cnt:
        #     reward += 4 * next_zero_util_cnt
        # elif zero_util_cnt > next_zero_util_cnt:
        #     reward -= 4 * zero_util_cnt
        # # 만약, 한쪽으로 SFC가 몰린다면, Reward를 추가한다.
        # # 반대라면, reward를 감소 시킨다.
        # sfc_cnt_in_same_srv = self._get_sfc_cnt_in_same_srv(state)
        # next_sfc_cnt_in_same_srv = self._get_sfc_cnt_in_same_srv(next_state)
        # if sfc_cnt_in_same_srv < next_sfc_cnt_in_same_srv:
        #     reward += 4 * next_sfc_cnt_in_same_srv
        # elif sfc_cnt_in_same_srv > next_sfc_cnt_in_same_srv:
        #     reward -= 4 * sfc_cnt_in_same_srv
        # if not is_moved:
        #     return - 5 * (len(state.srvs) - zero_util_cnt + len(state.sfcs) - sfc_cnt_in_same_srv)
        
        if not is_moved:
            return -100
        srv_len = len(state.srvs)
        sfc_len = len(state.sfcs)
        zero_util_cnt = self._get_zero_util_cnt(state)
        next_zero_util_cnt = self._get_zero_util_cnt(next_state)
        sfc_cnt_in_same_srv = self._get_sfc_cnt_in_same_srv(state)
        next_sfc_cnt_in_same_srv = self._get_sfc_cnt_in_same_srv(next_state)
        reward = (next_zero_util_cnt + next_sfc_cnt_in_same_srv) ** 5
        return reward

    def _get_zero_util_cnt(self, state: State) -> int:
        cnt = 0
        for srv in state.srvs:
            if len(srv.vnfs) == 0:
                cnt += 1
        return cnt
    
    def _get_sfc_cnt_in_same_srv(self, state: State) -> int:
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

    def _get_state(self) -> State:
        state = deepcopy(State(
            edge=self._get_edge(),
            srvs=self._get_srvs(),
            vnfs=self._get_vnfs(),
            sfcs=self._get_sfcs(),
        ))
        return state
