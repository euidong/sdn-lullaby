class Agent:
    def __init__(self) -> None:
        self.memory = []
        self.q_table = {}

    def decide_action(self, state) -> int:
        pass

    def update(self, state, action, reward, next_state) -> None:
        pass

    # save q_table
    def save(self) -> None:
        pass

    # load q_table
    def load(self) -> None:
        pass
