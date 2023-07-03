import time
from typing import List, Dict

import torch
import numpy as np
import torch.nn as nn

from src.model.dqn import NN
from src.env import Environment
from src.api.simulator import Simulator
from src.memory.replay import ReplayMemory
from src.dataType import State, Action, Scene
from src.utils import get_zero_util_cnt, get_sfc_cnt_in_same_srv


class DQNAgent:
    MAX_MEMORY_LEN = 10_000
    BATCH_SIZE = 32
    vm_selection_input_num = 11
    vm_placement_input_num = 11

    def __init__(self, srv_num: int, init_epsilon: float = 0.5, final_epsilon: float = 0.1, vnf_s_lr: float = 0.01, vnf_p_lr: float = 0.01, alpha=0.1, gamma=1.0) -> None:
        self.srv_num = srv_num
        self.epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.memory = ReplayMemory(self.BATCH_SIZE, self.MAX_MEMORY_LEN)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.vm_selection_model = NN(
            self.vm_selection_input_num, 1, num_layers=4).to(self.device)
        self.vm_placement_model = NN(
            self.vm_placement_input_num, 1, num_layers=4).to(self.device)
        self.vm_selection_optimizer = torch.optim.Adam(
            self.vm_selection_model.parameters(), lr=vnf_s_lr)
        self.vm_placement_optimzer = torch.optim.Adam(
            self.vm_placement_model.parameters(), lr=vnf_p_lr)
        self.loss_fn = nn.HuberLoss()
        self.prev_vnf_num = None

    def decide_action(self, state: State, epsilon_sub: float) -> Action:
        possible_actions = self._get_possible_actions(state)
        vm_s_in = self._convert_state_to_vm_selection_input(state)
        vnf_num = len(state.vnfs)
        epsilon = max(self.final_epsilon, self.epsilon - epsilon_sub)
        is_random = np.random.uniform() < epsilon
        if is_random:
            possible_actions
            vnf_idxs = []
            for i in range(len(state.vnfs)):
                if len(possible_actions[i]) > 0: vnf_idxs.append(i)
            vm_s_out = torch.tensor(np.random.choice(vnf_idxs, 1))
        else:
            self.vm_selection_model.eval()
            with torch.no_grad():
                vm_s_out = self.vm_selection_model(
                    vm_s_in.unsqueeze(0))[0]
                vm_s_out = vm_s_out * torch.tensor([True if len(possible_actions[i]) > 0 else False for i in range(len(state.vnfs))]).to(self.device)
                vm_s_out = vm_s_out.max(1)[1]
        vm_p_in = self._convert_state_to_vm_placement_input(state, int(vm_s_out))
        if is_random:
            srv_idxs = []
            for i in range(len(state.srvs)):
                if i in possible_actions[int(vm_s_out)]: srv_idxs.append(i)
            vm_p_out = torch.tensor(np.random.choice(srv_idxs, 1))
        else:
            self.vm_placement_model.eval()
            with torch.no_grad():
                vm_p_out = self.vm_placement_model(
                    vm_p_in.unsqueeze(0))[0]
                vm_p_out = vm_p_out * torch.tensor([True if i in possible_actions[int(vm_s_out)] else False for i in range(len(state.srvs))]).to(self.device)
                vm_p_out = vm_p_out.max(1)[1]
        scene = Scene(
            vm_s_in=vm_s_in,
            vm_s_out=vm_s_out,
            vm_p_in=vm_p_in,
            vm_p_out=vm_p_out,
            reward=None,  # this data will get from the env
            next_vm_p_in=None,  # this data will get from the env
            next_vm_s_in=None,  # this data will get from the env
        )

        if self.prev_vnf_num == self.prev_vnf_num and vnf_num in self.memory.memory:
            self.memory.memory[vnf_num][-1].next_vm_s_in = vm_s_in
            self.memory.memory[vnf_num][-1].next_vm_p_in = vm_p_in
        self.memory.append(vnf_num, scene)
        self.prev_vnf_num = vnf_num
        return Action(
            vnf_id=int(vm_s_out),
            srv_id=int(vm_p_out),
        )

    def update(self, _state, _action, reward, _next_state) -> None:
        self.memory.memory[self.prev_vnf_num][-1].reward = reward
        # sample a minibatch from memory
        scene_batch = self.memory.sample()
        if len(scene_batch) < self.BATCH_SIZE:
            return
        vm_s_in_batch = torch.stack(
            [scene.vm_s_in for scene in scene_batch]).to(self.device)
        vm_s_out_batch = torch.tensor(
            [scene.vm_s_out for scene in scene_batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        vm_p_in_batch = torch.stack(
            [scene.vm_p_in for scene in scene_batch]).to(self.device)
        vm_p_out_batch = torch.tensor(
            [scene.vm_p_out for scene in scene_batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(
            [scene.reward for scene in scene_batch]).unsqueeze(1).to(self.device)
        next_vm_s_in_batch = torch.stack(
            [scene.next_vm_s_in for scene in scene_batch]).to(self.device)
        next_vm_p_in_batch = torch.stack(
            [scene.next_vm_p_in for scene in scene_batch]).to(self.device)

        # set model to eval mode
        self.vm_selection_model.eval()
        self.vm_placement_model.eval()
        # get state-action value
        vm_selection_q = self.vm_selection_model(
            vm_s_in_batch)[0].gather(1, vm_s_out_batch)
        vm_placement_q = self.vm_placement_model(
            vm_p_in_batch)[0].gather(1, vm_p_out_batch)

        # calculate next_state-max_action value
        vm_selection_expect_q = reward_batch + self.gamma * \
            self.vm_selection_model(next_vm_s_in_batch)[0].max(1)[
                0].detach().unsqueeze(1)
        vm_placement_expect_q = reward_batch + self.gamma * \
            self.vm_placement_model(next_vm_p_in_batch)[0].max(1)[
                0].detach().unsqueeze(1)

        # set model to train mode
        self.vm_selection_model.train()

        # loss = distance between state-action value and next_state-max-action * gamma + reward
        vm_selection_loss = self.loss_fn(
            vm_selection_q, vm_selection_expect_q)

        # update model
        self.vm_selection_optimizer.zero_grad()
        vm_selection_loss.backward()
        self.vm_selection_optimizer.step()

        # set model to train mode
        self.vm_placement_model.train()

        # loss = distance between state-action value and next_state-max-action * gamma + reward
        vm_placement_loss = self.loss_fn(
            vm_placement_q, vm_placement_expect_q)

        # update model
        self.vm_placement_optimzer.zero_grad()
        vm_placement_loss.backward()
        self.vm_placement_optimzer.step()

    def save(self) -> None:
        torch.save(self.vm_selection_model.state_dict(),
                   "param/vm_selection_model.pth")
        torch.save(self.vm_placement_model.state_dict(),
                   "param/vm_placement_model.pth")

    def load(self) -> None:
        self.vm_selection_model.load_state_dict(
            torch.load("param/vm_selection_model.pth"))
        self.vm_selection_model.eval()
        self.vm_placement_model.load_state_dict(
            torch.load("param/vm_placement_model.pth"))
        self.vm_placement_model.eval()

    def _convert_state_to_vm_selection_input(self, state: State) -> torch.Tensor:
        vnf_num = len(state.vnfs)
        vm_selection_input = torch.zeros(vnf_num, self.vm_selection_input_num)
        for vnf in state.vnfs:
            vm_selection_input[vnf.id] = torch.tensor([
                vnf.cpu_req, vnf.mem_req, vnf.sfc_id,
                state.srvs[vnf.srv_id].cpu_cap, state.srvs[vnf.srv_id].mem_cap,
                state.srvs[vnf.srv_id].cpu_load, state.srvs[vnf.srv_id].mem_load,
                state.edge.cpu_cap, state.edge.mem_cap,
                state.edge.cpu_load, state.edge.mem_load,
            ])
        return vm_selection_input.to(self.device)

    def _convert_state_to_vm_placement_input(self, state: State, vm_id: int) -> torch.Tensor:
        vm_placement_input = torch.zeros(
            self.srv_num, self.vm_placement_input_num)
        for srv in state.srvs:
            vm_placement_input[srv.id] = torch.tensor([
                srv.cpu_cap, srv.mem_cap, srv.cpu_load, srv.mem_load,
                state.vnfs[vm_id].cpu_req, state.vnfs[vm_id].mem_req, state.vnfs[vm_id].sfc_id,
                state.edge.cpu_cap, state.edge.mem_cap, state.edge.cpu_load, state.edge.mem_load
            ])
        return vm_placement_input.to(self.device)

    def _get_possible_actions(self, state: State) -> Dict[int, List[int]]:
        '''return possible actions for each state

        Args:
            state (State): state

        Returns:
            Dict[int, List[int]]: possible actions
                                     ex) {vnfId: [srvId1, srvId2, ...], vnfId2: [srvId1, srvId2, ...], ...}
        '''
        possible_actions = {}
        for vnf in state.vnfs:
            possible_actions[vnf.id] = []
            for srv in state.srvs:
                # 동일한 srv로 다시 전송하는 것 방지
                if vnf.srv_id == srv.id: continue
                # capacity 확인
                if srv.cpu_cap - srv.cpu_load < vnf.cpu_req or srv.mem_cap - srv.mem_load < vnf.mem_req: continue
                possible_actions[vnf.id].append(srv.id)
        return possible_actions

def train():
    max_episode_num = 1_000
    debug_every_episode = 100
    max_vnf_num = 30
    sfc_n = 8
    srv_n = 16
    srv_cpu_cap = 16
    srv_mem_cap = 96
    max_edge_load = 0.3
    init_epsilon = 1.0
    final_epsilon = 0.1
    vnf_s_lr = 5e-4
    vnf_p_lr = 5e-4
    api = Simulator(srv_n,
                    srv_cpu_cap,
                    srv_mem_cap,
                    max_vnf_num,
                    sfc_n,
                    max_edge_load)
    env = Environment(api)
    agent = DQNAgent(srv_num=srv_n, init_epsilon=init_epsilon, final_epsilon=final_epsilon, vnf_s_lr=vnf_s_lr, vnf_p_lr=vnf_p_lr)
    training_start = time.time()

    ch_slp_srv = []
    init_slp_srv = []
    final_slp_srv = []
    ch_sfc_in_same_srv = []
    init_sfc_in_same_srv = []
    final_sfc_in_same_srv = []

    for episode in range(1, max_episode_num + 1):
        history = []
        state = env.reset()
        init_value = {
            "zero_util_cnt": get_zero_util_cnt(state),
            "sfc_cnt_in_same_srv": get_sfc_cnt_in_same_srv(state),
        }
        max_episode_len = env.max_episode_steps
        for step in range(max_episode_len):
            action = agent.decide_action(state, epsilon_sub=episode / max_episode_num)
            history.append((state, action))
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break
        result_value = {
            "zero_util_cnt": get_zero_util_cnt(state),
            "sfc_cnt_in_same_srv": get_sfc_cnt_in_same_srv(state),
        }

        ch_slp_srv.append(
            result_value["zero_util_cnt"] - init_value["zero_util_cnt"])
        init_slp_srv.append(init_value["zero_util_cnt"])
        final_slp_srv.append(result_value["zero_util_cnt"])
        ch_sfc_in_same_srv.append(
            result_value["sfc_cnt_in_same_srv"] - init_value["sfc_cnt_in_same_srv"])
        init_sfc_in_same_srv.append(init_value["sfc_cnt_in_same_srv"])
        final_sfc_in_same_srv.append(result_value["sfc_cnt_in_same_srv"])

        mean_100_change_slp_srv = np.mean(ch_slp_srv[-100:])
        meam_100_init_slp_srv = np.mean(init_slp_srv[-100:])
        mean_100_final_slp_srv = np.mean(final_slp_srv[-100:])
        mean_100_change_sfc_in_same_srv = np.mean(ch_sfc_in_same_srv[-100:])
        meam_100_init_sfc_in_same_srv = np.mean(init_sfc_in_same_srv[-100:])
        mean_100_final_sfc_in_same_srv = np.mean(final_sfc_in_same_srv[-100:])
        timestamp = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - training_start))

        debug_msg = "[{}] Episode {:04}, Step {:04}, #SleepSrv ({:02.3f})({:02.3f}->{:02.3f}/{}), #SFCinSameSrv ({:02.3f})({:02.3f}->{:02.3f}/{})".format(
            timestamp, episode, step,
            mean_100_change_slp_srv, meam_100_init_slp_srv, mean_100_final_slp_srv, srv_n,
            mean_100_change_sfc_in_same_srv, meam_100_init_sfc_in_same_srv, mean_100_final_sfc_in_same_srv, sfc_n,
        )
        print(debug_msg, end='\r', flush=True)
        history.append((state, None))
        if episode % debug_every_episode == 0:
            print('\x1b[2K' + debug_msg, flush=True)
            # animator = Animator(srv_n=srv_n, sfc_n=sfc_n, vnf_n=max_vnf_num,
            #                     srv_mem_cap=srv_mem_cap, srv_cpu_cap=srv_cpu_cap, history=history)
            # animator.save(f'./result/episode{episode}.mp4')
    agent.save()

if __name__ == '__main__':
    train()
