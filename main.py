import time

import numpy as np

from agent.dqn import DQNAgent as Agent
from env import Environment
from api.simulator import Simulator
from animator.animator import Animator
from utils import get_zero_util_cnt, get_sfc_cnt_in_same_srv


def main():
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
    agent = Agent(srv_num=srv_n, init_epsilon=init_epsilon, final_epsilon=final_epsilon, vnf_s_lr=vnf_s_lr, vnf_p_lr=vnf_p_lr)
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


if __name__ == "__main__":
    main()
