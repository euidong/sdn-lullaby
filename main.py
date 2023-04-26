from agent import Agent
from env import Environment
from api.simulator import Simulator
from converter import Converter


def main():
    max_episode_num = 100
    max_episode_len = 1000
    max_vnf_num = 100
    srv_n = 4
    srv_cpu_cap = 8
    srv_mem_cap = 32
    api = Simulator(srv_n, srv_cpu_cap, srv_mem_cap, max_vnf_num)
    env = Environment(api)
    agent = Agent()

    for i in range(max_episode_num):
        s = env.reset()
        e_s = Converter.encode_state(s)
        print(f"Episode {i} is started")
        for j in range(max_episode_len):
            e_a = agent.decide_action(e_s, duration=i*j)
            a = Converter.decode_action(e_a)
            ns, r, done = env.step(a)
            e_ns = Converter.encode_state(ns)
            agent.update(e_s, e_a, r, e_ns)
            e_s = e_ns
            if done:
                break
            if r != 0:
                print(f"Episode {i} step {j}'s reward is {r}.")
        print(
            f"sleeping server: {env.get_zero_util_cnt()} / {env.get_srv_cnt()}")
    agent.save()


if __name__ == "__main__":
    main()
