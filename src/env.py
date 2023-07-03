import random
from copy import deepcopy
from typing import List, Tuple

import torch
import numpy as np
import torch.multiprocessing as mp

from src.api.api import Api
from src.dataType import Edge, Server, VNF, SFC, State, Action
from src.utils import get_zero_util_cnt, get_sfc_cnt_in_same_srv


class Environment:
    def __init__(self, api: Api, seed: int = 927) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.api = api

    # return next_state, reward, done
    def step(self, action: Action) -> Tuple[State, int, bool]:
        self._elapsed_steps += 1
        done = self._elapsed_steps >= self.max_episode_steps
        state = self._get_state()
        is_moved = self._move_vnf(action.vnf_id, action.srv_id)
        next_state = self._get_state()
        reward = self._calc_reward(is_moved, state, next_state, done)
        return (next_state, reward, done)

    def reset(self) -> State:
        self.api.reset()
        self._elapsed_steps = 0
        self.init_state = self._get_state()
        self.max_episode_steps = len(self.init_state.vnfs)
        return self.init_state

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

    def _calc_reward(self, is_moved: bool, state: State, next_state: State, done: bool) -> int:
        if not done:
            return 0
        srv_n = len(state.srvs)
        sfc_n = len(state.sfcs)
        next_zero_util_cnt = get_zero_util_cnt(next_state)
        next_sfc_cnt_in_same_srv = get_sfc_cnt_in_same_srv(next_state)
        reward = next_zero_util_cnt / srv_n + next_sfc_cnt_in_same_srv / sfc_n
        return reward

    def _get_state(self) -> State:
        state = deepcopy(State(
            edge=self._get_edge(),
            srvs=self._get_srvs(),
            vnfs=self._get_vnfs(),
            sfcs=self._get_sfcs(),
        ))
        return state

class MultiprocessEnvironment:
    def __init__(self, seed, n_workers, make_env_fn):
        self.seed = seed
        self.make_env_fn = make_env_fn
        self.n_workers = n_workers
        # for message passing (send, recv)
        # [0] parent_process_end
        # [1] child_process_end
        self.pipes = [mp.Pipe() for _ in range(n_workers)]
        # make process with rank and message passing pipe
        self.workers = [mp.Process(target=self._work, args=(rank, self.pipes[rank][1])) for rank in range(n_workers)]
        [w.start() for w in self.workers]

    def _close(self, **kwargs):
        self._broadcast_msg(('close', kwargs))
        [w.join() for w in self.workers]

    def _send_msg(self, msg, rank):
        parent_end = self.pipes[rank][0]
        parent_end.send(msg)
    
    def _broadcast_msg(self, msg):
        [self._send_msg(msg, rank) for rank in range(self.n_workers)]

    def _work(self, rank, child_end):
        env = self.make_env_fn(self.seed + rank)
        while True:
            cmd, kwargs = child_end.recv()
            if cmd == 'step':
                child_end.send(env.step(**kwargs))
            elif cmd == 'reset':
                child_end.send(env.reset(**kwargs))
            elif cmd == 'close':
                del env
                child_end.close()
                break
            else:
                del env
                child_end.close()
                break

    def reset(self, ranks=None, **kwargs):
        if ranks is not None:
            [self._send_msg(('reset', kwargs), rank) for rank in ranks]
            return [self.pipes[rank][0].recv() for rank in ranks]
        else:
            self._broadcast_msg(('reset', kwargs))
            return [self.pipes[rank][0].recv() for rank in range(self.n_workers)]

    def step(self, actions):
        assert len(actions) == self.n_workers

        [self._send_msg(('step', {'action': actions[rank]}), rank) for rank in range(self.n_workers)]
        next_states, rewards, dones = [], [], []
        for rank in range(self.n_workers):
            next_state, reward, done = self.pipes[rank][0].recv()
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return next_states, rewards, dones
