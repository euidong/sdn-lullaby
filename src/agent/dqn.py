import os
import time
from dataclasses import dataclass
from typing import List, Dict, Callable

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from src.model.dqn import DQNValueInfo, DQNValue
from src.env import Environment
from src.api.simulator import Simulator
from src.memory.replay import ReplayMemory
from src.dataType import State, Action, Scene
from src.utils import (
    DebugInfo,
    print_debug_info,
    get_device,
    save_animation,
    get_zero_util_cnt,
    get_sfc_cnt_in_same_srv,
    get_possible_actions,
    convert_state_to_vnf_selection_input,
    convert_state_to_vnf_placement_input,
)
from src.const import VNF_PLACEMENT_IN_DIM, VNF_SELECTION_IN_DIM


@dataclass
class DQNAgentInfo:
    srv_n: int
    max_vnf_num: int
    init_epsilon: float
    final_epsilon: float
    vnf_s_lr: float
    vnf_p_lr: float
    gamma: float
    vnf_s_model_info: DQNValueInfo
    vnf_p_model_info: DQNValueInfo


class DQNAgent:
    MAX_MEMORY_LEN = 1_000
    BATCH_SIZE = 32
    vnf_selection_input_num = VNF_SELECTION_IN_DIM
    vnf_placement_input_num = VNF_PLACEMENT_IN_DIM

    def __init__(self, info: DQNAgentInfo) -> None:
        self.info = info
        self.device = info.vnf_s_model_info.device

        self.memory = ReplayMemory(self.BATCH_SIZE, self.MAX_MEMORY_LEN)

        self.vnf_selection_model = DQNValue(info.vnf_s_model_info)
        self.vnf_placement_model = DQNValue(info.vnf_p_model_info)
        self.vnf_selection_optimizer = torch.optim.Adam(
            self.vnf_selection_model.parameters(), lr=info.vnf_s_lr)
        self.vnf_placement_optimzer = torch.optim.Adam(
            self.vnf_placement_model.parameters(), lr=info.vnf_p_lr)
        self.loss_fn = nn.HuberLoss()

    def get_exploration_rate(self, epsilon_sub: float) -> float:
        return max(self.info.final_epsilon, self.info.init_epsilon - epsilon_sub)

    def decide_action(self, state: State, epsilon_sub: float) -> Action:
        possible_actions = get_possible_actions(state, self.info.max_vnf_num)
        vnf_s_in = convert_state_to_vnf_selection_input(
            state, self.info.max_vnf_num)
        epsilon = self.get_exploration_rate(epsilon_sub)
        is_random = np.random.uniform() < epsilon
        if is_random:
            vnf_idxs = []
            for i in range(len(state.vnfs)):
                if len(possible_actions[i]) > 0:
                    vnf_idxs.append(i)
            vnf_s_out = torch.tensor(np.random.choice(vnf_idxs, 1))
        else:
            self.vnf_selection_model.eval()
            with torch.no_grad():
                vnf_s_out = self.vnf_selection_model(vnf_s_in.unsqueeze(0))
                vnf_s_out = vnf_s_out + \
                    torch.tensor([0 if len(possible_actions[i]) > 0 else -
                                 torch.inf for i in range(len(possible_actions))]).to(self.device)
                vnf_s_out = vnf_s_out.max(1)[1]
        vnf_p_in = convert_state_to_vnf_placement_input(state, int(vnf_s_out))
        if is_random:
            srv_idxs = []
            for i in range(len(state.srvs)):
                if i in possible_actions[int(vnf_s_out)]:
                    srv_idxs.append(i)
            vnf_p_out = torch.tensor(np.random.choice(srv_idxs, 1))
        else:
            self.vnf_placement_model.eval()
            with torch.no_grad():
                vnf_p_out = self.vnf_placement_model(vnf_p_in.unsqueeze(0))
                vnf_p_out = vnf_p_out + torch.tensor([0 if i in possible_actions[int(
                    vnf_s_out)] else -torch.inf for i in range(len(state.srvs))]).to(self.device)
                vnf_p_out = vnf_p_out.max(1)[1]
        scene = Scene(
            vnf_s_in=vnf_s_in,
            vnf_s_out=vnf_s_out,
            vnf_p_in=vnf_p_in,
            vnf_p_out=vnf_p_out,
            reward=None,  # this data will get from the env
            next_vnf_p_in=None,  # this data will get from the env
            next_vnf_s_in=None,  # this data will get from the env
        )

        if len(self.memory) > 0:
            self.memory.buffer[-1].next_vnf_s_in = vnf_s_in
            self.memory.buffer[-1].next_vnf_p_in = vnf_p_in
        self.memory.append(scene)
        return Action(
            vnf_id=int(vnf_s_out),
            srv_id=int(vnf_p_out),
        )

    def update(self, _state, _action, reward, _next_state) -> None:
        self.memory.buffer[-1].reward = reward
        # sample a minibatch from memory
        scene_batch = self.memory.sample()
        if len(scene_batch) < self.BATCH_SIZE:
            return
        vnf_s_in_batch = torch.stack(
            [scene.vnf_s_in for scene in scene_batch]).to(self.device)
        vnf_s_out_batch = torch.tensor(
            [scene.vnf_s_out for scene in scene_batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        vnf_p_in_batch = torch.stack(
            [scene.vnf_p_in for scene in scene_batch]).to(self.device)
        vnf_p_out_batch = torch.tensor(
            [scene.vnf_p_out for scene in scene_batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(
            [scene.reward for scene in scene_batch]).unsqueeze(1).to(self.device)
        next_vnf_s_in_batch = torch.stack(
            [scene.next_vnf_s_in for scene in scene_batch]).to(self.device)
        next_vnf_p_in_batch = torch.stack(
            [scene.next_vnf_p_in for scene in scene_batch]).to(self.device)

        # set model to eval mode
        self.vnf_selection_model.eval()
        self.vnf_placement_model.eval()
        # get state-action value
        vnf_selection_q = self.vnf_selection_model(
            vnf_s_in_batch).gather(1, vnf_s_out_batch)
        vnf_placement_q = self.vnf_placement_model(
            vnf_p_in_batch).gather(1, vnf_p_out_batch)

        # calculate next_state-max_action value
        vnf_selection_expect_q = reward_batch + self.info.gamma * \
            self.vnf_selection_model(next_vnf_s_in_batch).max(1)[
                0].detach().unsqueeze(1)
        vnf_placement_expect_q = reward_batch + self.info.gamma * \
            self.vnf_placement_model(next_vnf_p_in_batch).max(1)[
                0].detach().unsqueeze(1)

        # set model to train mode
        self.vnf_selection_model.train()

        # loss = distance between state-action value and next_state-max-action * gamma + reward
        vnf_selection_loss = self.loss_fn(
            vnf_selection_q, vnf_selection_expect_q)

        # update model
        self.vnf_selection_optimizer.zero_grad()
        vnf_selection_loss.backward()
        self.vnf_selection_optimizer.step()

        # set model to train mode
        self.vnf_placement_model.train()

        # loss = distance between state-action value and next_state-max-action * gamma + reward
        vnf_placement_loss = self.loss_fn(
            vnf_placement_q, vnf_placement_expect_q)

        # update model
        self.vnf_placement_optimzer.zero_grad()
        vnf_placement_loss.backward()
        self.vnf_placement_optimzer.step()

    def save(self) -> None:
        os.makedirs("param/dqn", exist_ok=True)
        torch.save(self.vnf_selection_model.state_dict(),
                   "param/dqn/vnf_selection_model.pth")
        torch.save(self.vnf_placement_model.state_dict(),
                   "param/dqn/vnf_placement_model.pth")

    def load(self) -> None:
        self.vnf_selection_model.load_state_dict(
            torch.load("param/dqn/vnf_selection_model.pth"))
        self.vnf_placement_model.load_state_dict(
            torch.load("param/dqn/vnf_placement_model.pth"))


@dataclass
class TrainArgs:
    srv_n: int
    sfc_n: int
    max_vnf_num: int
    seed: int
    max_episode_num: int
    debug_every_n_episode: int
    evaluate_every_n_episode: int


def train(agent: DQNAgent, make_env_fn: Callable, args: TrainArgs, file_name_prefix: str):
    env = make_env_fn(args.seed)
    training_start = time.time()

    ch_slp_srv = []
    init_slp_srv = []
    final_slp_srv = []
    ch_sfc_in_same_srv = []
    init_sfc_in_same_srv = []
    final_sfc_in_same_srv = []
    steps = []
    rewards = []
    explorations = []
    debug_infos = []

    for episode in range(1, args.max_episode_num + 1):
        history = []
        state = env.reset()
        init_value = {
            "zero_util_cnt": get_zero_util_cnt(state),
            "sfc_cnt_in_same_srv": get_sfc_cnt_in_same_srv(state),
        }
        max_episode_len = env.max_episode_steps
        epsilon_sub = (episode / args.max_episode_num) * 0.5
        for step in range(max_episode_len):
            action = agent.decide_action(state, epsilon_sub)
            history.append((state, action))
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break
        rewards.append(reward)
        steps.append(step)
        explorations.append(agent.get_exploration_rate(epsilon_sub))
        final_value = {
            "zero_util_cnt": get_zero_util_cnt(state),
            "sfc_cnt_in_same_srv": get_sfc_cnt_in_same_srv(state),
        }
        ch_slp_srv.append(
            final_value["zero_util_cnt"] - init_value["zero_util_cnt"])
        init_slp_srv.append(init_value["zero_util_cnt"])
        final_slp_srv.append(final_value["zero_util_cnt"])
        ch_sfc_in_same_srv.append(
            final_value["sfc_cnt_in_same_srv"] - init_value["sfc_cnt_in_same_srv"])
        init_sfc_in_same_srv.append(init_value["sfc_cnt_in_same_srv"])
        final_sfc_in_same_srv.append(final_value["sfc_cnt_in_same_srv"])

        debug_info = DebugInfo(
            timestamp=time.strftime("%H:%M:%S", time.gmtime(
                time.time() - training_start)),
            episode=episode,
            mean_100_step=np.mean(steps[-100:]),
            std_100_step=np.std(steps[-100:]),
            mean_100_init_slp_srv=np.mean(init_slp_srv[-100:]),
            std_100_init_slp_srv=np.std(init_slp_srv[-100:]),
            mean_100_final_slp_srv=np.mean(final_slp_srv[-100:]),
            std_100_final_slp_srv=np.std(final_slp_srv[-100:]),
            mean_100_change_slp_srv=np.mean(ch_slp_srv[-100:]),
            std_100_change_slp_srv=np.std(ch_slp_srv[-100:]),
            srv_n=srv_n,
            mean_100_init_sfc_in_same_srv=np.mean(init_sfc_in_same_srv[-100:]),
            std_100_init_sfc_in_same_srv=np.std(init_sfc_in_same_srv[-100:]),
            mean_100_final_sfc_in_same_srv=np.mean(
                final_sfc_in_same_srv[-100:]),
            std_100_final_sfc_in_same_srv=np.std(final_sfc_in_same_srv[-100:]),
            mean_100_change_sfc_in_same_srv=np.mean(ch_sfc_in_same_srv[-100:]),
            std_100_change_sfc_in_same_srv=np.std(ch_sfc_in_same_srv[-100:]),
            sfc_n=sfc_n,
            mean_100_exploration=np.mean(explorations[-100:]),
            std_100_exploration=np.std(explorations[-100:]),
            mean_100_reward=np.mean(rewards[-100:]),
            std_100_reward=np.std(rewards[-100:]),
        )
        debug_infos.append(debug_info)
        print_debug_info(debug_info, refresh=False)
        history.append((state, None))
        if episode % args.debug_every_n_episode == 0:
            print_debug_info(debug_info, refresh=True)
        if episode % args.evaluate_every_n_episode == 0:
            evaluate(agent, make_env_fn, seed=args.seed,
                     file_name=f'{file_name_prefix}_episode{episode}')

    pd.DataFrame(debug_infos).to_csv(
        f'result/dqn/{file_name_prefix}_debug_info.csv', index=False)


def evaluate(agent: DQNAgent, make_env_fn: Callable, seed: int = 927, file_name: str = 'test'):
    env = make_env_fn(seed)
    state = env.reset()
    history = []
    while True:
        action = agent.decide_action(state, 0)
        history.append((state, action))
        state, reward, done = env.step(action)
        agent.memory.buffer[-1].reward = reward
        if done:
            break
    history.append((state, None))
    os.makedirs('./result/dqn', exist_ok=True)
    save_animation(
        srv_n=srv_n, sfc_n=sfc_n, vnf_n=max_vnf_num,
        srv_mem_cap=srv_mem_cap, srv_cpu_cap=srv_cpu_cap,
        history=history, path=f'./result/dqn/{file_name}.mp4',
    )


if __name__ == '__main__':
    # Simulator Args
    srv_n = 8
    sfc_n = 8
    max_vnf_num = 20
    srv_cpu_cap = 32
    srv_mem_cap = 96
    # max_edge_load = 0.1
    seed = 927

    def make_env_fn(seed): return Environment(
        api=Simulator(srv_n, srv_cpu_cap, srv_mem_cap,
                      max_vnf_num, sfc_n, max_edge_load),
        seed=seed,
    )
    max_edge_loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for max_edge_load in max_edge_loads:
        device = get_device()
        agent_info = DQNAgentInfo(
            srv_n=srv_n,
            max_vnf_num=max_vnf_num,
            init_epsilon=0.5,
            final_epsilon=0.0,
            vnf_s_lr=1e-3,
            vnf_p_lr=1e-3,
            gamma=0.99,
            vnf_s_model_info=DQNValueInfo(
                in_dim=VNF_SELECTION_IN_DIM,
                hidden_dim=32,
                num_heads=4,
                num_blocks=4,
                device=device,
            ),
            vnf_p_model_info=DQNValueInfo(
                in_dim=VNF_PLACEMENT_IN_DIM,
                hidden_dim=32,
                num_heads=4,
                num_blocks=4,
                device=device,
            ),
        )
        agent = DQNAgent(agent_info)

        train_args = TrainArgs(
            srv_n=srv_n,
            sfc_n=sfc_n,
            max_vnf_num=max_vnf_num,
            seed=seed,
            max_episode_num=10_000,
            debug_every_n_episode=500,
            evaluate_every_n_episode=500
        )

        evaluate(agent, make_env_fn, seed=seed,
                 file_name=f'edge_load={max_edge_load}_init')
        train(agent, make_env_fn, train_args,
              file_name_prefix=f'edge_load={max_edge_load}')
        evaluate(agent, make_env_fn, seed=seed,
                 file_name=f'edge_load={max_edge_load}_final')

        agent.save()
