from agent.dqn import DQNAgent as Agent
from env import Environment
from api.simulator import Simulator
from animator.animator import Animator


def main():
    max_episode_num = 1000
    max_vnf_num = 10
    sfc_n = 4
    srv_n = 4
    srv_cpu_cap = 8
    srv_mem_cap = 32
    max_edge_load = 0.3
    api = Simulator(srv_n,
                    srv_cpu_cap,
                    srv_mem_cap,
                    max_vnf_num,
                    sfc_n,
                    max_edge_load)
    env = Environment(api)
    agent = Agent(srv_num=srv_n)
    for i in range(max_episode_num):
        history = []
        state = env.reset()
        print(f"Episode {i} is started")
        print(f"sleeping server: {env._get_zero_util_cnt(state)} / {srv_n}")
        print(
            f"sfc in same server: {env._get_sfc_cnt_in_same_srv(state)} / {sfc_n}")
        print(f"first vnfs: {state.vnfs}")
        max_episode_len = len(state.vnfs) * srv_n
        for j in range(max_episode_len):
            action = agent.decide_action(state, duration=i // 10 + 1)
            history.append((state, action))
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break
            if reward != 0:
                print(f"Episode {i} step {j}'s reward is {reward}.")
        history.append((state, None))

        print(f"sleeping server: {env._get_zero_util_cnt(state)} / {srv_n}")
        print(
            f"sfc in same server: {env._get_sfc_cnt_in_same_srv(state)} / {sfc_n}")
        if i % 50 == 0:
            animator = Animator(srv_n=srv_n, sfc_n=sfc_n, vnf_n=max_vnf_num,
                                srv_mem_cap=srv_mem_cap, srv_cpu_cap=srv_cpu_cap, history=history)
            animator.save(f'./result/episode{i}.mp4')
    agent.save()


if __name__ == "__main__":
    main()
