import os
import time
import torch
from dataclasses import dataclass

from dataType import State, Action
from api.simulator import Simulator
from agent.memory import EpisodeMemory
from env import Environment, MultiprocessEnvironment
from const import VNF_SELECTION_IN_DIM, VNF_PLACEMENT_IN_DIM
from agent.model import PPOPolicyInfo, PPOValueInfo, PPOPolicy, PPOValue
from utils import (
    get_device,
    setup_mp_env,
    logit_to_prob,
    print_debug_info,
    get_possible_actions,
    convert_state_to_vnf_selection_input,
    convert_state_to_vnf_placement_input,
)

@dataclass
class PPOAgentInfo:
    vnf_s_policy_info: PPOPolicyInfo
    vnf_p_policy_info: PPOPolicyInfo
    vnf_s_value_info: PPOValueInfo
    vnf_p_value_info: PPOValueInfo
    vnf_s_policy_lr: float
    vnf_p_policy_lr: float
    vnf_s_value_lr: float
    vnf_p_value_lr: float
    vnf_s_policy_clip_range: float
    vnf_p_policy_clip_range: float
    vnf_s_entropy_loss_weight: float
    vnf_p_entropy_loss_weight: float
    vnf_s_policy_max_grad_norm: float
    vnf_p_policy_max_grad_norm: float
    vnf_s_value_clip_range: float
    vnf_p_value_clip_range: float
    vnf_s_value_max_grad_norm: float
    vnf_p_value_max_grad_norm: float


class PPOAgent:
    def __init__(self, info: PPOAgentInfo):
        self.info = info

        self.vnf_s_policy = PPOPolicy(info.vnf_s_policy_info)
        self.vnf_p_policy = PPOPolicy(info.vnf_p_policy_info)
        self.vnf_s_value = PPOValue(info.vnf_s_value_info)
        self.vnf_p_value = PPOValue(info.vnf_p_value_info)

        self.vnf_s_policy_optimizer = torch.optim.Adam(self.vnf_s_policy.parameters(), lr=info.vnf_s_policy_lr)
        self.vnf_p_policy_optimizer = torch.optim.Adam(self.vnf_p_policy.parameters(), lr=info.vnf_p_policy_lr)
        self.vnf_s_value_optimizer = torch.optim.Adam(self.vnf_s_value.parameters(), lr=info.vnf_s_value_lr)
        self.vnf_p_value_optimizer = torch.optim.Adam(self.vnf_p_value.parameters(), lr=info.vnf_p_value_lr)


    def decide_action(self, state: State, greedy: bool=True)-> Action:
        p_actions = get_possible_actions(state, self.info.vnf_s_policy_info.out_dim)
        if greedy: # Greedy
            vnf_s_in = convert_state_to_vnf_selection_input(state, self.info.vnf_s_policy_info.out_dim)
            vnf_s_out = self.vnf_s_policy(vnf_s_in)
            vnf_s_out = vnf_s_out * torch.tensor([True if len(p_actions[i]) > 0 else False for i in range(len(state.vnfs))])
            vnf_id = int(vnf_s_out.max(1)[1])
            vnf_p_in = convert_state_to_vnf_placement_input(state, vnf_id)
            vnf_p_out = self.vnf_p_policy(vnf_p_in)
            vnf_p_out = vnf_p_out * torch.tensor([True if i in p_actions[vnf_id] else False for i in range(len(state.srvs))])
            srv_id = int(vnf_p_out.max(1)[1])
        else: # Stochastic
            vnf_s_in = convert_state_to_vnf_selection_input(state, self.info.vnf_s_policy_info.out_dim)
            vnf_s_out = self.vnf_s_policy(vnf_s_in)
            vnf_s_out = vnf_s_out * torch.tensor([True if len(p_actions[i]) > 0 else False for i in range(len(state.vnfs))])
            vnf_s_probs = logit_to_prob(vnf_s_out)
            dist = torch.distributions.Categorical(probs=vnf_s_probs)
            vnf_id = int(dist.sample())
            vnf_p_in = convert_state_to_vnf_placement_input(state, vnf_id)
            vnf_p_out = self.vnf_p_policy(vnf_p_in)
            vnf_p_out = vnf_p_out * torch.tensor([True if i in p_actions[vnf_id] else False for i in range(len(state.srvs))])
            vnf_p_probs = logit_to_prob(vnf_p_out)
            dist = torch.distributions.Categorical(probs=vnf_p_probs)
            srv_id = int(dist.sample())
        return Action(vnf_id=vnf_id, srv_id=srv_id)

    def update_policy(self, vnf_s_ins: torch.Tensor, vnf_p_ins: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, gaes: torch.Tensor, logpas: torch.Tensor, values: torch.Tensor) -> None:
        # TODO: Possible Action 고려안했는데 생각해볼만하다.
        actions = actions.to(self.info.vnf_s_policy_info.device)
        logpas = logpas.to(self.info.vnf_s_policy_info.device)
        gaes = gaes.to(self.info.vnf_s_policy_info.device)

        vnf_s_outs = self.vnf_s_policy(vnf_s_ins)
        vnf_s_probs = logit_to_prob(vnf_s_outs)
        vnf_s_dist = torch.distributions.Categorical(probs=vnf_s_probs)
        vnf_s_logpas_pred = vnf_s_dist.log_prob(actions[:, 0])
        vnf_s_entropies_pred = vnf_s_dist.entropy()

        vnf_s_rations = torch.exp(vnf_s_logpas_pred - logpas[:, 0])
        vnf_s_pi_obj = vnf_s_rations * gaes[:, 0]
        vnf_s_pi_obj_clipped = vnf_s_rations.clamp(1.0 - self.info.vnf_s_policy_clip_range, 1.0 + self.info.vnf_s_policy_clip_range) * gaes[:, 0]

        vnf_s_policy_loss = -torch.min(vnf_s_pi_obj, vnf_s_pi_obj_clipped).mean()
        vnf_s_entropy_loss = -vnf_s_entropies_pred.mean()
        vnf_s_policy_loss = vnf_s_policy_loss + self.info.vnf_s_entropy_loss_weight * vnf_s_entropy_loss

        self.vnf_s_policy_optimizer.zero_grad()
        vnf_s_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vnf_s_policy.parameters(), self.info.vnf_s_policy_max_grad_norm)
        self.vnf_s_policy_optimizer.step()

        vnf_p_outs = self.vnf_p_policy(vnf_p_ins)
        vnf_p_probs = logit_to_prob(vnf_p_outs)
        vnf_p_dist = torch.distributions.Categorical(probs=vnf_p_probs)
        vnf_p_logpas_pred = vnf_p_dist.log_prob(actions[:, 1])
        vnf_p_entropies_pred = vnf_p_dist.entropy()

        vnf_p_rations = torch.exp(vnf_p_logpas_pred - logpas[:, 1])
        vnf_p_pi_obj = vnf_p_rations * gaes[:, 1]
        vnf_p_pi_obj_clipped = vnf_p_rations.clamp(1.0 - self.info.vnf_p_policy_clip_range, 1.0 + self.info.vnf_p_policy_clip_range) * gaes[:, 1]

        vnf_p_policy_loss = -torch.min(vnf_p_pi_obj, vnf_p_pi_obj_clipped).mean()
        vnf_p_entropy_loss = -vnf_p_entropies_pred.mean()
        vnf_p_policy_loss = vnf_p_policy_loss + self.info.vnf_p_entropy_loss_weight * vnf_p_entropy_loss

        self.vnf_p_policy_optimizer.zero_grad()
        vnf_p_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vnf_p_policy.parameters(), self.info.vnf_p_policy_max_grad_norm)
        self.vnf_p_policy_optimizer.step()

    def update_value(self, vnf_s_ins: torch.Tensor, vnf_p_ins: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, gaes: torch.Tensor, logpas: torch.Tensor, values: torch.Tensor) -> None:
        vnf_s_returns = returns[:, 0].to(self.info.vnf_s_value_info.device)
        vnf_p_returns = returns[:, 1].to(self.info.vnf_p_value_info.device)
        vnf_s_values = values[:, 0].to(self.info.vnf_s_value_info.device)
        vnf_p_values = values[:, 1].to(self.info.vnf_p_value_info.device)

        vnf_s_value_pred = self.vnf_s_value(vnf_s_ins)
        vnf_s_value_pred_clipped = vnf_s_values + (vnf_s_value_pred - vnf_s_values).clamp(-self.info.vnf_s_value_clip_range, self.info.vnf_s_value_clip_range)

        vnf_s_value_loss = (vnf_s_value_pred - vnf_s_returns).pow(2)
        vnf_s_value_loss_clipped = (vnf_s_value_pred_clipped - vnf_s_returns).pow(2)
        vnf_s_value_loss = 0.5 * torch.max(vnf_s_value_loss, vnf_s_value_loss_clipped).mean()

        self.vnf_s_value_optimizer.zero_grad()
        vnf_s_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vnf_s_value.parameters(), self.info.vnf_s_value_max_grad_norm)
        self.vnf_s_value_optimizer.step()

        vnf_p_value_pred = self.vnf_p_value(vnf_p_ins)
        vnf_p_value_pred_clipped = vnf_p_values + (vnf_p_value_pred - vnf_p_values).clamp(-self.info.vnf_p_value_clip_range, self.info.vnf_p_value_clip_range)

        vnf_p_value_loss = (vnf_p_value_pred - vnf_p_returns).pow(2)
        vnf_p_value_loss_clipped = (vnf_p_value_pred_clipped - vnf_p_returns).pow(2)
        vnf_p_value_loss = 0.5 * torch.max(vnf_p_value_loss, vnf_p_value_loss_clipped).mean()

        self.vnf_p_value_optimizer.zero_grad()
        vnf_p_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vnf_p_value.parameters(), self.info.vnf_p_value_max_grad_norm)
        self.vnf_p_value_optimizer.step()

    def get_logpas_pred(self, vnf_s_ins: torch.Tensor, vnf_p_ins: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.to(self.info.vnf_s_policy_info.device)
        with torch.no_grad():
            vnf_s_logits = self.vnf_s_policy(vnf_s_ins)
            vnf_s_probs = logit_to_prob(vnf_s_logits)
            vnf_s_dist = torch.distributions.Categorical(probs=vnf_s_probs)
            vnf_s_logpas_pred = vnf_s_dist.log_prob(actions[:, 0])

            vnf_p_logits = self.vnf_p_policy(vnf_p_ins)
            vnf_p_probs = logit_to_prob(vnf_p_logits)
            vnf_p_dist = torch.distributions.Categorical(probs=vnf_p_probs)
            vnf_p_logpas_pred = vnf_p_dist.log_prob(actions[:, 1])

        return torch.stack([vnf_s_logpas_pred, vnf_p_logpas_pred], dim=1)


    def save(self) -> None:
        os.makedirs('./result/model', exist_ok=True)
        torch.save(self.vnf_p_policy.state_dict(), './result/model/vnf_p_policy.pt')
        torch.save(self.vnf_s_policy.state_dict(), './result/model/vnf_s_policy.pt')

    def load(self) -> None:
        self.vnf_p_policy.load_state_dict(torch.load('./result/model/vnf_p_policy.pt'))
        self.vnf_s_policy.load_state_dict(torch.load('./result/model/vnf_s_policy.pt'))


def train():
    setup_mp_env()

    # Simulator Args
    max_vnf_num = 30
    srv_n = 8
    sfc_n = 16
    srv_cpu_cap = 32
    srv_mem_cap = 96
    max_edge_load = 0.3
    seed = 927
    
    make_env_fn = lambda seed : Environment(
        api=Simulator(srv_n, srv_cpu_cap, srv_mem_cap, max_vnf_num, sfc_n, max_edge_load),
        seed=seed,
    )

    device = get_device()
    agent_info = PPOAgentInfo(
        vnf_s_policy_info=PPOPolicyInfo(
            in_dim=VNF_SELECTION_IN_DIM,
            hidden_dim=32,
            out_dim=max_vnf_num,
            num_blocks=2,
            num_heads=2,
            device=device,
        ),
        vnf_p_policy_info=PPOPolicyInfo(
            in_dim=VNF_PLACEMENT_IN_DIM,
            hidden_dim=32,
            out_dim=srv_n,
            num_blocks=2,
            num_heads=2,
            device=device,
        ),
        vnf_s_value_info=PPOValueInfo(
            in_dim=VNF_SELECTION_IN_DIM,
            hidden_dim=32,
            seq_len=max_vnf_num,
            num_blocks=2,
            num_heads=2,
            device=device,
        ),
        vnf_p_value_info=PPOValueInfo(
            in_dim=VNF_PLACEMENT_IN_DIM,
            hidden_dim=32,
            seq_len=srv_n,
            num_blocks=2,
            num_heads=2,
            device=device,
        ),
        vnf_s_policy_lr=1e-3,
        vnf_p_policy_lr=1e-3,
        vnf_s_value_lr=1e-3,
        vnf_p_value_lr=1e-3,
        vnf_s_policy_clip_range=0.1,
        vnf_p_policy_clip_range=0.1,
        vnf_s_entropy_loss_weight=0.01,
        vnf_p_entropy_loss_weight=0.01,
        vnf_s_policy_max_grad_norm=float('inf'),
        vnf_p_policy_max_grad_norm=float('inf'),
        vnf_s_value_clip_range=float('inf'),
        vnf_p_value_clip_range=float('inf'),
        vnf_s_value_max_grad_norm=float('inf'),
        vnf_p_value_max_grad_norm=float('inf'),
    )
    agent = PPOAgent(agent_info)

    # Train Args
    memory_max_episode_num = 200
    max_episode_steps = max_vnf_num
    n_workers = 8
    batch_size = 32
    update_epochs = 80
    max_episode_num = 50_000
    policy_update_early_stop_threshold = 1e-3
    gamma = 0.99
    tau = 0.97
    
    memory = EpisodeMemory(
        n_workers, batch_size,
        gamma, tau,
        memory_max_episode_num, max_episode_steps,
        srv_n, sfc_n, max_vnf_num,
        VNF_SELECTION_IN_DIM, VNF_PLACEMENT_IN_DIM,
    )
    mp_env = MultiprocessEnvironment(seed, n_workers, make_env_fn)

    training_start = time.time()
    for episode in range(0, max_episode_num, memory_max_episode_num):
        memory.fill(mp_env, agent)
        debug_info = memory.get_debug_info(episode=episode, training_start=training_start)
        print_debug_info(debug_info, refresh=True)
        vnf_s_ins, vnf_p_ins, actions, _, _, logpas, _ = memory.samples(all=True)
        for _ in range(update_epochs):
            agent.update_policy(*memory.samples())
            logpas_pred = agent.get_logpas_pred(vnf_s_ins, vnf_p_ins, actions).cpu()
            early_stop = torch.abs(logpas - logpas_pred).mean() < policy_update_early_stop_threshold
            if early_stop:
                break
        for _ in range(update_epochs):
            agent.update_value(*memory.samples())
        memory.reset()
    agent.save()

if __name__ == '__main__':
    train()