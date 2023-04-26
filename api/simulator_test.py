import unittest
from api.simulator import Simulator
import numpy as np


class TestSimulator(unittest.TestCase):
    def test_reset_simulator(self):
        api = Simulator(srv_n=4, srv_cpu_cap=8, srv_mem_cap=32)
        api.reset()

        srvs = api.get_util_from_srvs()

        for srv in srvs:
            print(f'\nserver {srv.id}')
            print(f'cpu: {srv.cpu_cap} mem: {srv.mem_cap}')
            for vnf in srv.vnfs:
                print(
                    f'vnf {vnf.id} cpu: {vnf.cpu_req} mem: {vnf.mem_req} sfc: {vnf.sfc_id}')
            print('---------------------------------')
        edge_cpu_load, edge_mem_load = api._calc_edge_load()
        print(f'edge cpu: {edge_cpu_load} mem: {edge_mem_load}')
        self.assertGreaterEqual(max(edge_cpu_load, edge_mem_load), 0.5)

    def test_move_vnf(self):
        api = Simulator(srv_n=4, srv_cpu_cap=8, srv_mem_cap=32)
        api.reset()

        # calc vnf_cnt
        vnf_cnt = 0
        srvs = api.get_util_from_srvs()
        for srv in srvs:
            vnf_cnt += len(srv.vnfs)

        # moving success test
        is_moved = False
        while not is_moved:
            srvs = api.get_util_from_srvs()
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
        srvs = api.get_util_from_srvs()
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
        srvs = api.get_util_from_srvs()
        vnf_id = np.random.choice(vnf_cnt)
        srv_id = np.random.choice(len(srvs))
        srvs[srv_id].cpu_cap = 0
        srvs[srv_id].mem_cap = 0
        is_moved = api.move_vnf(vnf_id, srv_id)
        if is_moved:
            self.fail('server capacity is over')