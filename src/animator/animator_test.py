import random
import unittest

from src.env import Environment
from src.dataType import Action
from src.api.simulator import Simulator
from src.animator.animator import Animator


class TestAnimator(unittest.TestCase):
    def test_gen_animation(self):
        srv_cpu_cap = 8
        srv_mem_cap = 32
        api = Simulator(srv_cpu_cap=srv_cpu_cap, srv_mem_cap=srv_mem_cap)
        env = Environment(api)

        state = env.reset()
        srv_n = len(state.srvs)
        vnf_n = len(state.vnfs)
        sfc_n = len(state.sfcs)

        history = []
        animator = Animator(srv_n=srv_n, sfc_n=sfc_n, vnf_n=vnf_n,
                            srv_mem_cap=srv_mem_cap, srv_cpu_cap=srv_cpu_cap, history=history)
        for i in range(10):
            action = Action(random.randint(0, vnf_n-1),
                            random.randint(0, srv_n-1))
            history.append((state, action))
            state, _, _ = env.step(action)
        history.append((state, None))
        animator.save('test.mp4')
        self.assertTrue(True)
