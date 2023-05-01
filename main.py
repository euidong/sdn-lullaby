from agent.dqn import DQNAgent as Agent
from env import Environment
from api.simulator import Simulator


def main():
    max_episode_num = 400
    max_episode_len = 250
    max_vnf_num = 10
    srv_n = 4
    srv_cpu_cap = 4
    srv_mem_cap = 16
    api = Simulator(srv_n, srv_cpu_cap, srv_mem_cap, max_vnf_num)
    env = Environment(api)
    agent = Agent(vnf_num=max_vnf_num, srv_num=srv_n)

    for i in range(max_episode_num):
        state = env.reset()
        print(f"Episode {i+1} is started")
        print(f"sleeping server: {env._get_zero_util_cnt(state)} / {srv_n}")
        print(f"first vnfs: {state.vnfs}")
        for j in range(max_episode_len):
            action = agent.decide_action(state, duration=(i + 1))
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break
            if reward > 0:
                print(f"Episode {i+1} step {j+1}'s reward is {reward}.")
        print(f"sleeping server: {env._get_zero_util_cnt(state)} / {srv_n}")
    agent.save()


if __name__ == "__main__":
    main()
