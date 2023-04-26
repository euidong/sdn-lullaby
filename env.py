from typing import List, Tuple
from api.api import Api

class Environment:
    def __init__(self, api: Api) -> None:
        self.api = api
        pass

    # return next_state, reward, done
    def step(self, action) -> Tuple[int, int, False]:
        """TODO: move vnf and return next_state, reward, done"""
        pass

    def reset(self) -> int:
        """TODO: reset env and return state"""
        pass


    def _moveVNF(self, vnfId, srvId) -> None:
        return self.api.move_vnf(vnfId, srvId)

    def _calcReward(self) -> int:
        srvs = self._getSrvs()
        reward = 0
        """TODO: calc reward"""
        return reward

    def _getSrvs(self) -> List[Tuple[int, int]]:
        return self.api.get_util_from_srvs()
