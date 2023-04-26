from typing import List, Tuple
from api.api import Api
from dataType import Server


class Environment:
    def __init__(self, api: Api) -> None:
        self.api = api

    # return next_state, reward, done
    def step(self, action) -> Tuple[int, int, bool]:
        """TODO: move vnf and return next_state, reward, done"""
        state = self._get_state()
        is_moved = self._move_vnf(action[0], action[1])
        next_state = self._get_state()
        reward = self._calc_reward(is_moved, state, next_state)
        done = False
        return (next_state, reward, done)

    def reset(self) -> int:
        self.api.reset()
        return self._get_state()

    def _move_vnf(self, vnfId, srvId) -> None:
        return self.api.move_vnf(vnfId, srvId)

    def _get_srvs(self) -> List[Server]:
        return self.api.get_util_from_srvs()

    def _calc_reward(self, is_moved: bool, state: List[Server], next_state: List[Server]) -> int:
        if not is_moved:
            return 0
        reward = 1
        zero_util_cnt = self._get_zero_util_cnt(state)
        next_zero_util_cnt = self._get_zero_util_cnt(next_state)
        # 만약, 특정 server의 전원이 꺼졌다면, reward를 더 준다.
        if zero_util_cnt > next_zero_util_cnt:
            reward += 2
        # 반대로, 특정 server의 전원을 킨다면, reward를 감소 시킨다.
        elif zero_util_cnt < next_zero_util_cnt:
            reward -= 2
        # TODO: 한쪽으로 SFC가 몰린다면, Reward를 추가한다.
        return reward

    def _get_zero_util_cnt(self, state: List[Server]) -> int:
        cnt = 0
        for srv in state:
            if len(srv.vnfs) == 0:
                cnt += 1
        return cnt

    def _get_state(self) -> int:
        """TODO: define state"""
        return self._get_srvs()
