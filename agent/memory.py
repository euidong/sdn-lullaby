import gc
import time
from collections import deque
from typing import List, Optional
from copy import deepcopy

import numpy as np
import torch
import itertools

from const import EPS
from dataType import State, Action
from env import MultiprocessEnvironment
from const import VNF_SELECTION_IN_DIM, VNF_PLACEMENT_IN_DIM
from utils import (
    DebugInfo,
    get_zero_util_cnt,
    get_possible_actions,
    get_info_from_logits,
    get_sfc_cnt_in_same_srv,
    convert_state_to_vnf_selection_input,
    convert_state_to_vnf_placement_input,
)

class Memory:
    def __init__(self, batch_size, max_memory_len=100_000):
        self.batch_size = batch_size
        self.max_memory_len = max_memory_len
        self.memory = {}

    def __len__(self) -> int:
        return len(self.memory.keys())

    def sample(self, vnf_no: Optional[int] = None) -> List[any]:
        if not vnf_no:
            vnf_no = np.random.choice(list(self.memory.keys()))
        if vnf_no in self.memory:
            return []
        if len(self.memory[vnf_no]) < self.batch_size:
            return []
        return np.random.choice(list(itertools.islice(
            self.memory[vnf_no], 0, len(self.memory[vnf_no]) - 1)), self.batch_size)

    def append(self, vnf_no: int, data: any) -> None:
        if not vnf_no in self.memory:
            self.memory[vnf_no] = deque(maxlen=self.max_memory_len)
        self.memory[vnf_no].append(data)

    def last(self, vnf_no: int, n: int) -> List[any]:
        return list(itertools.islice(self.memory[vnf_no], max(0, len(self.memory)-n), len(self.memory)))

class EpisodeMemory:
    def __init__(self, 
                 n_workers: int, batch_size: int,
                 gamma: float, tau: float,
                 memory_max_episode_num: int, max_episode_steps: int,
                 srv_n: int, max_sfc_n: int, max_vnf_num: int,
                 vnf_s_in_dim: int = VNF_SELECTION_IN_DIM, vnf_p_in_dim: int = VNF_PLACEMENT_IN_DIM,
                ):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.memory_max_episode_num = memory_max_episode_num
        self.max_episode_steps = max_episode_steps
        self.srv_n = srv_n
        self.max_sfc_n = max_sfc_n
        self.max_vnf_num = max_vnf_num
        self.vnf_s_in_dim = vnf_s_in_dim
        self.vnf_p_in_dim = vnf_p_in_dim

        self.discounts = torch.logspace(0, max_episode_steps+1, steps=max_episode_steps+1, base=gamma, dtype=torch.float64)
        self.tau_discounts = torch.logspace(0, max_episode_steps+1, steps=max_episode_steps+1, base=(gamma*tau), dtype=torch.float64)

        self.reset()

    def reset(self) -> None:
        self._clear_debug_info()
        self._clear_memory()
        gc.collect()

    def fill(self, mp_env: MultiprocessEnvironment, agent) -> None:
        workers_rewards = torch.zeros((self.n_workers, self.max_episode_steps), dtype=torch.float32)
        workers_exploratory = torch.zeros((self.n_workers, self.max_episode_steps, 2), dtype=torch.float32)
        workers_steps = torch.zeros((self.n_workers), dtype=torch.int32)
        workers_seconds = torch.tensor([time.time(),] * self.n_workers, dtype=torch.float64)

        states = mp_env.reset()
        init_states = deepcopy(states)

        while len(self.episode_steps[self.episode_steps > 0]) < self.memory_max_episode_num / 2:
            p_actions = [get_possible_actions(state, self.max_vnf_num) for state in states]
            with torch.no_grad():
                vnf_s_ins = torch.stack([convert_state_to_vnf_selection_input(state, self.max_vnf_num) for state in states], dim=0)
                vnf_s_outs = agent.vnf_s_policy(vnf_s_ins).detach().cpu()
                vnf_s_outs = vnf_s_outs * torch.tensor([[True if len(p_actions[i][vnf_id]) > 0 else False for vnf_id in range(self.max_vnf_num)] for i in range(self.n_workers)])
                vnf_s_actions, vnf_s_logpas, vnf_s_is_exploratory = get_info_from_logits(vnf_s_outs)
                vnf_p_ins = torch.stack([convert_state_to_vnf_placement_input(state, vnf_s_action) for state, vnf_s_action in zip(states, vnf_s_actions)], dim=0)
                vnf_p_outs = agent.vnf_p_policy(vnf_p_ins).detach().cpu()
                vnf_p_outs = vnf_p_outs * torch.tensor([[True if srv_id in p_actions[i][int(vnf_s_actions[i])] else False for srv_id in range(self.srv_n)] for i in range(self.n_workers)])
                vnf_p_actions, vnf_p_logpas, vnf_p_is_exploratory = get_info_from_logits(vnf_p_outs)
            actions = [Action(vnf_id=vnf_s_actions[i], srv_id=vnf_p_actions[i]) for i in range(self.n_workers)]
            next_states, rewards, dones = mp_env.step(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.bool)

            self.vnf_s_ins[self.cur_episode_idxs, workers_steps] = vnf_s_ins
            self.vnf_p_ins[self.cur_episode_idxs, workers_steps] = vnf_p_ins
            self.actions[self.cur_episode_idxs, workers_steps] = torch.stack([vnf_s_actions, vnf_p_actions], dim=1)
            self.logpas[self.cur_episode_idxs, workers_steps] = torch.stack([vnf_s_logpas, vnf_p_logpas], dim=1)

            workers_exploratory[torch.arange(self.n_workers), workers_steps] = torch.stack([vnf_s_is_exploratory.to(torch.float32), vnf_p_is_exploratory], dim=1)
            workers_rewards[torch.arange(self.n_workers), workers_steps] = rewards

            if dones.sum() > 0:
                idx_done = torch.where(dones)[0]
                next_values = torch.zeros((self.n_workers, 2))
                next_p_actions = [get_possible_actions(next_state, self.max_vnf_num) for next_state in next_states]
                with torch.no_grad():
                    next_vnf_s_ins = torch.stack([convert_state_to_vnf_selection_input(next_state, self.max_vnf_num) for next_state in next_states], dim=0)
                    vnf_s_values = agent.vnf_s_value(next_vnf_s_ins).detach().cpu()
                    vnf_s_logits = agent.vnf_s_policy(next_vnf_s_ins).detach().cpu()
                    vnf_s_logits = vnf_s_logits * torch.tensor([[True if len(next_p_actions[i][vnf_id]) > 0 else False for vnf_id in range(self.max_vnf_num)] for i in range(self.n_workers)])
                    vnf_s_actions = []
                    for vnf_s_logit in vnf_s_logits:
                        vnf_s_actions.append(torch.argmax(vnf_s_logit[vnf_s_logit != 0]))
                    vnf_s_actions = torch.tensor(vnf_s_actions, dtype=torch.int32)
                    next_vnf_p_ins = torch.stack([convert_state_to_vnf_placement_input(next_state, vnf_s_action) for next_state, vnf_s_action in zip(next_states, vnf_s_actions)], dim=0)
                    vnf_p_values = agent.vnf_p_value(next_vnf_p_ins).detach().cpu()
                    next_values[idx_done] = torch.stack([vnf_s_values, vnf_p_values], dim=1)[idx_done]
            states = next_states
            workers_steps += 1

            if dones.sum() > 0:
                done_init_states = [init_states[i] for i in idx_done]
                done_final_states = [states[i] for i in idx_done]
                self._save_debug_info(done_init_states, done_final_states)
                new_states = mp_env.reset(ranks=idx_done)
                for new_s_idx, s_idx in enumerate(idx_done):
                    states[s_idx] = new_states[new_s_idx]
                init_states[s_idx] = deepcopy(states[s_idx])

                for w_idx in range(self.n_workers):
                    if w_idx not in idx_done: continue

                    e_idx = self.cur_episode_idxs[w_idx]
                    T = workers_steps[w_idx]
                    self.episode_steps[e_idx] = T
                    self.episode_rewards[e_idx] = workers_rewards[w_idx, :T].sum()
                    self.episode_explorations[e_idx] = workers_exploratory[w_idx, :T].mean()
                    self.episode_seconds[e_idx] = time.time() - workers_seconds[w_idx]

                    vnf_s_ep_rewards = torch.concat([workers_rewards[w_idx, :T], next_values[w_idx][0].unsqueeze(0)], dim=0)
                    vnf_p_ep_rewards = torch.concat([workers_rewards[w_idx, :T], next_values[w_idx][1].unsqueeze(0)], dim=0)
                    ep_discounts = self.discounts[:T+1]
                    vnf_s_ep_returns = torch.Tensor([(ep_discounts[:T+1-t] * vnf_s_ep_rewards[t:]).sum() for t in range(T)])
                    vnf_p_ep_returns = torch.Tensor([(ep_discounts[:T+1-t] * vnf_p_ep_rewards[t:]).sum() for t in range(T)])
                    self.returns[e_idx, :T] = torch.stack([vnf_s_ep_returns, vnf_p_ep_returns], dim=1)

                    ep_vnf_s_ins = self.vnf_s_ins[e_idx, :T]
                    ep_vnf_p_ins = self.vnf_p_ins[e_idx, :T]

                    with torch.no_grad():
                        vnf_s_ep_values = torch.cat([agent.vnf_s_value(ep_vnf_s_ins).detach().cpu(), next_values[w_idx][0].unsqueeze(0)])
                        vnf_p_ep_values = torch.cat([agent.vnf_p_value(ep_vnf_p_ins).detach().cpu(), next_values[w_idx][1].unsqueeze(0)])
                        
                    vnf_s_ep_deltas = vnf_s_ep_rewards[:-1] + self.gamma * vnf_s_ep_values[1:] - vnf_s_ep_values[:-1]
                    vnf_s_ep_gaes = torch.Tensor([(self.tau_discounts[:T-t] * vnf_s_ep_deltas[t:]).sum() for t in range(T)])

                    vnf_p_ep_deltas = vnf_p_ep_rewards[:-1] + self.gamma * vnf_p_ep_values[1:] - vnf_p_ep_values[:-1]
                    vnf_p_ep_gaes = torch.Tensor([(self.tau_discounts[:T-t] * vnf_p_ep_deltas[t:]).sum() for t in range(T)])

                    self.gaes[e_idx, :T] = torch.stack([vnf_s_ep_gaes, vnf_p_ep_gaes], dim=1)

                    workers_exploratory[w_idx, :] = False
                    workers_rewards[w_idx, :] = 0
                    workers_steps[w_idx] = 0
                    workers_seconds[w_idx] = time.time()

                    new_ep_id = max(self.cur_episode_idxs) + 1
                    if new_ep_id >= self.memory_max_episode_num:
                        break
                    self.cur_episode_idxs[w_idx] = new_ep_id
        ep_idxs = self.episode_steps > 0
        ep_t = self.episode_steps[ep_idxs]
        
        self.vnf_s_ins = torch.concat([row[:t] for row, t in zip(self.vnf_s_ins[ep_idxs], ep_t)])
        self.vnf_p_ins = torch.concat([row[:t] for row, t in zip(self.vnf_p_ins[ep_idxs], ep_t)])
        self.actions = torch.concat([row[:t] for row, t in zip(self.actions[ep_idxs], ep_t)])
        self.returns = torch.concat([row[:t] for row, t in zip(self.returns[ep_idxs], ep_t)])
        self.gaes = torch.concat([row[:t] for row, t in zip(self.gaes[ep_idxs], ep_t)])
        self.logpas = torch.concat([row[:t] for row, t in zip(self.logpas[ep_idxs], ep_t)])
        self.values = torch.stack([agent.vnf_s_value(self.vnf_s_ins).detach().cpu(), agent.vnf_p_value(self.vnf_p_ins).detach().cpu()], dim=1) # 확인 필요

        ep_r = self.episode_rewards[ep_idxs]
        ep_x = self.episode_explorations[ep_idxs]
        ep_s = self.episode_seconds[ep_idxs]
        
        return ep_t, ep_r, ep_x, ep_s

    def samples(self, all: bool = False):
        if all:
            return self.vnf_s_ins, self.vnf_p_ins, self.actions, self.returns, self.gaes, self.logpas, self.values
        mem_size = len(self)
        batch_idxs = np.random.choice(mem_size, self.batch_size, replace=False)
        vnf_s_ins = self.vnf_s_ins[batch_idxs]
        vnf_p_ins = self.vnf_p_ins[batch_idxs]
        actions = self.actions[batch_idxs]
        returns = self.returns[batch_idxs]
        gaes = self.gaes[batch_idxs]
        gaes = (gaes - gaes.mean()) / (gaes.std() + EPS)
        logpas = self.logpas[batch_idxs]
        values = self.values[batch_idxs]

        return vnf_s_ins, vnf_p_ins, actions, returns, gaes, logpas, values

    def get_debug_info(self, episode:int, training_start: float) -> DebugInfo:
        mean_100_change_slp_srv = np.mean(self.ch_slp_srv[-100:])
        meam_100_init_slp_srv = np.mean(self.init_slp_srv[-100:])
        mean_100_final_slp_srv = np.mean(self.final_slp_srv[-100:])
        mean_100_change_sfc_in_same_srv = np.mean(self.ch_sfc_in_same_srv[-100:])
        meam_100_init_sfc_in_same_srv = np.mean(self.init_sfc_in_same_srv[-100:])
        mean_100_final_sfc_in_same_srv = np.mean(self.final_sfc_in_same_srv[-100:])
        mean_100_steps = self.episode_steps[self.episode_steps > 0][-100:].to(torch.float32).mean()
        mean_100_exploration = self.episode_explorations[self.episode_explorations > 0][-100:].to(torch.float32).mean()

        timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))

        return DebugInfo(
            timestamp=timestamp,
            episode=episode,
            step=mean_100_steps,
            mean_100_change_slp_srv=mean_100_change_slp_srv,
            mean_100_init_slp_srv=meam_100_init_slp_srv,
            mean_100_final_slp_srv=mean_100_final_slp_srv,
            srv_n=self.srv_n,
            mean_100_change_sfc_in_same_srv=mean_100_change_sfc_in_same_srv,
            mean_100_init_sfc_in_same_srv=meam_100_init_sfc_in_same_srv,
            mean_100_final_sfc_in_same_srv=mean_100_final_sfc_in_same_srv,
            sfc_n=self.max_sfc_n,
            mean_100_exploration=mean_100_exploration,
        )

    def __len__(self):
        return len(self.actions)

    def _clear_debug_info(self) -> None:
        self.ch_slp_srv = []
        self.init_slp_srv = []
        self.final_slp_srv = []
        self.ch_sfc_in_same_srv = []
        self.init_sfc_in_same_srv = []
        self.final_sfc_in_same_srv = []
    
    def _save_debug_info(self, init_states: List[State], final_states: List[State]):
        for init_state, final_state in zip(init_states, final_states):
            init_zero_util_cnt = get_zero_util_cnt(init_state)
            final_zero_util_cnt = get_zero_util_cnt(final_state)
            init_sfc_cnt_in_same_srv = get_sfc_cnt_in_same_srv(init_state)
            final_sfc_cnt_in_same_srv = get_sfc_cnt_in_same_srv(final_state)

            self.init_slp_srv.append(init_zero_util_cnt)
            self.final_slp_srv.append(final_zero_util_cnt)
            self.ch_slp_srv.append(final_zero_util_cnt - init_zero_util_cnt)

            self.init_sfc_in_same_srv.append(init_sfc_cnt_in_same_srv)
            self.final_sfc_in_same_srv.append(final_sfc_cnt_in_same_srv)
            self.ch_sfc_in_same_srv.append(final_sfc_cnt_in_same_srv - init_sfc_cnt_in_same_srv)

    def _clear_memory(self) -> None:
        self.vnf_s_ins = torch.empty((self.memory_max_episode_num, self.max_episode_steps, self.max_vnf_num, self.vnf_s_in_dim), dtype=torch.float32)
        self.vnf_p_ins = torch.empty((self.memory_max_episode_num, self.max_episode_steps, self.srv_n, self.vnf_p_in_dim), dtype=torch.float32)
        self.actions = torch.empty((self.memory_max_episode_num, self.max_episode_steps, 2), dtype=torch.int32)
        self.returns = torch.empty((self.memory_max_episode_num, self.max_episode_steps, 2), dtype=torch.float32)
        self.values = torch.empty((self.memory_max_episode_num, self.max_episode_steps, 2), dtype=torch.float32)
        self.gaes = torch.empty((self.memory_max_episode_num, self.max_episode_steps, 2), dtype=torch.float32)
        self.logpas = torch.empty((self.memory_max_episode_num, self.max_episode_steps, 2), dtype=torch.float32)

        self.episode_steps = torch.zeros((self.memory_max_episode_num), dtype=torch.int32)
        self.episode_rewards = torch.zeros((self.memory_max_episode_num), dtype=torch.float32)
        self.episode_explorations = torch.zeros((self.memory_max_episode_num), dtype=torch.float32)
        self.episode_seconds = torch.zeros((self.memory_max_episode_num), dtype=torch.float64)

        self.cur_episode_idxs = torch.arange((self.n_workers), dtype=torch.int32)
        
