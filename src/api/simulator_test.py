import unittest

import numpy as np

from src.api.simulator import Simulator


class TestSimulator(unittest.TestCase):
    def test_reset_simulator(self):
        api = Simulator(srv_n=4, srv_cpu_cap=8, srv_mem_cap=32)
        util = 0.5
        sfc_n = 4
        api.reset(sfc_n, util)

        sfs = api.get_sfcs()
        if len(sfs) > sfc_n:
            self.fail('sfc count is too high')
        vnfs = api.get_vnfs()
        if len(vnfs) > api.max_vnf_num: 
            self.fail('vnf count is too high')
        edge = api.get_edge()
        edge_cpu_util = edge.cpu_load / edge.cpu_cap
        edge_mem_util = edge.mem_load / edge.mem_cap
        print(f'edge cpu util: {edge_cpu_util} mem util: {edge_mem_util}')
        if edge_cpu_util < util and edge_mem_util < util:
            self.fail('edge load is too low')

    def test_move_vnf(self):
        api = Simulator(srv_n=4, srv_cpu_cap=8, srv_mem_cap=32)
        api.reset()

        # calc vnf_cnt
        vnf_cnt = 0
        srvs = api.get_srvs()
        for srv in srvs:
            vnf_cnt += len(srv.vnfs)

        # moving success test
        is_moved = False
        while not is_moved:
            srvs = api.get_srvs()
            vnf_id = np.random.choice(vnf_cnt)
            # find current vnf's server id and vnf
            for srv in srvs:
                for vnf in srv.vnfs:
                    if vnf.id == vnf_id:
                        target_vnf = vnf
                        prev_vnf_srv_id = srv.id
                        break
            srv_id = np.random.choice(len(srvs))
            is_moved = api.move_vnf(vnf_id, srv_id)
            cur_vnf_srv_id = srv_id
        if target_vnf in srvs[prev_vnf_srv_id].vnfs:
            self.fail('vnf is not removed')
        if target_vnf not in srvs[cur_vnf_srv_id].vnfs:
            self.fail('vnf is not added')

        # moving fail test
        # 1. 없는 vnf_id
        vnf_id = vnf_cnt + 1
        srv_id = np.random.choice(len(srvs))
        is_moved = api.move_vnf(vnf_id, srv_id)
        if is_moved:
            self.fail('vnf is not exist')
        
        # 2. 없는 srv_id
        vnf_id = np.random.choice(vnf_cnt)
        srv_id = len(srvs) + 1
        is_moved = api.move_vnf(vnf_id, srv_id)
        if is_moved:
            self.fail('server is not exist')
        # 3. 이미 vnf_id를 가진 srv_id 전달
        vnf_id = np.random.choice(vnf_cnt)
        srvs = api.get_srvs()
        for srv in srvs:
            for vnf in srv.vnfs:
                if vnf.id == vnf_id:
                    target_vnf = vnf
                    srv_id = srv.id
                    break
        is_moved = api.move_vnf(vnf_id, srv_id)
        if is_moved:
            self.fail('vnf is already in server')
        
        # 4. capacity 초과
        srvs = api.get_srvs()
        vnf_id = np.random.choice(vnf_cnt)
        srv_id = np.random.choice(len(srvs))
        srvs[srv_id].cpu_cap = 0
        srvs[srv_id].mem_cap = 0
        is_moved = api.move_vnf(vnf_id, srv_id)
        if is_moved:
            self.fail('server capacity is over')
