import numpy as np
import torch
import torch.nn as nn
from agent.model import NN
from dataType import State, Action, Scene, Server
from agent.memory import Memory
from typing import List


class DQNAgent:
    MAX_MEMORY_LEN = 1000
    BATCH_SIZE = 32
    vm_selection_input_num = 11
    vm_placement_input_num = 11

    def __init__(self, srv_num: int, epsilon: float = 0.5, alpha=0.1, gamma=0.1) -> None:
        self.srv_num = srv_num
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.memory = Memory(self.BATCH_SIZE, self.MAX_MEMORY_LEN)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.vm_selection_model = NN(
            self.vm_selection_input_num, 1).to(self.device)
        self.vm_placement_model = NN(
            self.vm_placement_input_num, 1).to(self.device)
        self.vm_selection_optimizer = torch.optim.Adam(
            self.vm_selection_model.parameters(), lr=0.001)
        self.vm_placement_optimzer = torch.optim.Adam(
            self.vm_placement_model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.prev_vnf_num = None

    def decide_action(self, state: State, duration: int) -> Action:
        vm_s_in = self._convert_state_to_vm_selection_input(state)
        vnf_num = len(state.vnfs)
        is_random = np.random.uniform() < self.epsilon / duration
        if is_random:
            vm_s_out = torch.tensor([np.random.choice(len(state.vnfs))])
        else:
            self.vm_selection_model.eval()
            with torch.no_grad():
                vm_s_out = self.vm_selection_model(
                    vm_s_in.unsqueeze(0))[0].max(1)[1]
        vm_p_in = self._convert_state_to_vm_placement_input(
            state, int(vm_s_out))
        if is_random:
            vm_p_out = torch.tensor([np.random.choice(self.srv_num)])
        else:
            self.vm_placement_model.eval()
            with torch.no_grad():
                vm_p_out = self.vm_placement_model(
                    vm_p_in.unsqueeze(0))[0].max(1)[1]
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
                state.vnfs[vm_id].cpu_req, state.vnfs[vm_id].mem_req, state.vnfs[vm_id].sfc_id,
                srv.cpu_cap, srv.mem_cap, srv.cpu_load, srv.mem_load,
                state.edge.cpu_cap, state.edge.mem_cap, state.edge.cpu_load, state.edge.mem_load
            ])
        return vm_placement_input.to(self.device)

    # CHECK: version 2
    # Below code is version2 code.
    # def _convert_state_to_vm_placement_input(self, state: State, vm_id: int) -> torch.Tensor:
    #     vm_placement_input = torch.zeros(
    #         self.srv_num, self.vm_placement_input_num)
    #     for srv in state.srvs:
    #         if self._is_possible(srv, vm_id, state.vnfs[vm_id].cpu_req, state.vnfs[vm_id].mem_req):
    #             vm_placement_input[srv.id] = torch.tensor([
    #                 state.vnfs[vm_id].cpu_req, state.vnfs[vm_id].mem_req, state.vnfs[vm_id].sfc_id,
    #                 srv.cpu_cap, srv.mem_cap, srv.cpu_load, srv.mem_load,
    #                 state.edge.cpu_cap, state.edge.mem_cap, state.edge.cpu_load, state.edge.mem_load
    #             ])
    #         else:
    #             vm_placement_input[srv.id] = torch.zeros(self.vm_placement_input_num)
    #     return vm_placement_input.to(self.device)

    # def _is_possible(self, srv: Server, vm_id: int, vm_cpu_req, vm_mem_req) -> bool:
    #     if srv.cpu_cap < srv.cpu_load + vm_cpu_req:
    #         return False
    #     if srv.mem_cap < srv.mem_load + vm_mem_req:
    #         return False
    #     for vnf in srv.vnfs:
    #         if vnf.id == vm_id:
    #             return False
    #     return True
