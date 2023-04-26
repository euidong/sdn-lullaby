from agent import Agent
from env import Environment
from api.simulator import Simulator


def main():
    max_episode_num = 100
    api = Simulator()
    env = Environment(api)
    agent = Agent()

    for i in range(max_episode_num):
        state = env.reset()
        done = False
        while not done:
            action = agent.decide_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
    agent.save()


if __name__ == "__main__":
    main()
