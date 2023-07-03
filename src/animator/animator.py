from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter

from src.dataType import State, Action


class Animator:
    FPS = 4
    def __init__(self, srv_n, srv_cpu_cap, srv_mem_cap, sfc_n, vnf_n, history: List[Tuple[State, Optional[Action]]]):
        self.srv_n = srv_n
        self.sfc_n = sfc_n
        self.vnf_n = vnf_n
        self.srv_idxs = np.arange(srv_n)

        self.history = history
        fig, axs = plt.subplots(1, 2, figsize=(10, 12))
        self.fig = fig
        self.cmap = plt.get_cmap("Pastel1")

        axs[0].set_title('CPU')
        axs[0].set_xlabel('Server ID')
        axs[0].set_ylabel('# of CPU cores')

        axs[1].set_title('Memory')
        axs[1].set_xlabel('Server ID')
        axs[1].set_ylabel('Memory(GB)')

        self.cpu_cap_bar = axs[0].bar(self.srv_idxs, np.zeros(srv_n), fill=False,
                                      edgecolor='black', linestyle='--', linewidth=1)
        self.mem_cap_bar = axs[1].bar(self.srv_idxs, np.zeros(srv_n), fill=False,
                                      edgecolor='black', linestyle='--', linewidth=1)

        self.cpu_bars = [[axs[0].bar(self.srv_idxs, np.zeros(srv_n), color=self.cmap(
            i), edgecolor='black', linewidth=2) for _ in range(vnf_n)] for i in range(sfc_n)]
        self.mem_bars = [[axs[1].bar(self.srv_idxs, np.zeros(srv_n), color=self.cmap(
            i), edgecolor='black', linewidth=2) for _ in range(vnf_n)] for i in range(sfc_n)]
        self.cpu_bar_texts = [[[axs[0].text(0, 0, '', ha='center', va='center') for _ in range(srv_n)] for _ in range(vnf_n)] for _ in range(sfc_n)]
        self.mem_bar_texts = [[[axs[1].text(0, 0, '', ha='center', va='center') for _ in range(srv_n)] for _ in range(vnf_n)] for _ in range(sfc_n)]
        
        axs[0].xaxis.set_ticks(self.srv_idxs)
        axs[1].xaxis.set_ticks(self.srv_idxs)
        axs[0].yaxis.set_ticks(np.append(np.arange(0, srv_cpu_cap, srv_cpu_cap // 8),[srv_cpu_cap]))
        axs[1].yaxis.set_ticks(np.append(np.arange(0, srv_mem_cap, srv_mem_cap // 8),[srv_mem_cap]))

        handles = [mpatches.Patch(color=self.cmap(i), label=f'SFC{i+1}') for i in range(sfc_n)]
        self.legend = fig.legend(handles=handles)
        self.suptitle = fig.suptitle('', fontweight='bold')

        self.writer = FFMpegWriter(fps=self.FPS)

    def animate(self, i):
        state, action = self.history[i]
        self.draw_state(i, state, action)

    def save(self, path):
        anim = FuncAnimation(self.fig, self.animate,
                             frames=len(self.history), interval=1000/self.FPS)
        
        anim.save(path, writer=self.writer)

    def draw_state(self, attempt: int, state: State, action: Action = None) -> None:
        # define figure
        srv_n = self.srv_n

        srv_cpu_caps = np.array([srv.cpu_cap for srv in state.srvs])
        srv_mem_caps = np.array([srv.mem_cap for srv in state.srvs])

        cpu_bottom = np.zeros(srv_n)
        mem_bottom = np.zeros(srv_n)

        # 모든 bars/texts 초기화
        for k in range(srv_n):
            self.cpu_cap_bar[k].set_height(srv_cpu_caps[k])
            self.mem_cap_bar[k].set_height(srv_mem_caps[k])
        for i in range(self.sfc_n):
            for j in range(self.vnf_n):
                for k in range(srv_n):
                    self.cpu_bars[i][j][k].set_height(0)
                    self.mem_bars[i][j][k].set_height(0)
                    self.cpu_bar_texts[i][j][k].set_text('')
                    self.mem_bar_texts[i][j][k].set_text('')
        # 모든 bars 다시 그리기
        for i in range(self.sfc_n):
            for j in range(len(state.sfcs[i].vnfs)):
                cur_cpu_height = np.zeros(srv_n)
                cur_mem_height = np.zeros(srv_n)

                cur_cpu_height[state.sfcs[i].vnfs[j].srv_id] += state.sfcs[i].vnfs[j].cpu_req
                cur_mem_height[state.sfcs[i].vnfs[j].srv_id] += state.sfcs[i].vnfs[j].mem_req

                for k in range(srv_n):
                    self.cpu_bars[i][j][k].set_height(cur_cpu_height[k])
                    self.cpu_bars[i][j][k].set_y(cpu_bottom[k])
                    self.mem_bars[i][j][k].set_height(cur_mem_height[k])
                    self.mem_bars[i][j][k].set_y(mem_bottom[k])
                    cpu_bar = self.cpu_bars[i][j][k]
                    mem_bar = self.mem_bars[i][j][k]
                    cpu_height = cpu_bar.get_height()
                    mem_height = mem_bar.get_height()
                    if cpu_height > 0:
                        posx = cpu_bar.get_x() + cpu_bar.get_width() * 0.5
                        posy = cpu_height * 0.5 + cpu_bottom[k]
                        self.cpu_bar_texts[i][j][k].set_text(f'id:{state.sfcs[i].vnfs[j].id}\nreq:{int(cpu_height)}')
                        self.cpu_bar_texts[i][j][k].set_position((posx, posy))
                    if mem_height > 0:
                        posx = mem_bar.get_x() + mem_bar.get_width() * 0.5
                        posy = mem_height * 0.5 + mem_bottom[k]
                        self.mem_bar_texts[i][j][k].set_text(f'id:{state.sfcs[i].vnfs[j].id}\nreq:{int(cpu_height)}')
                        self.mem_bar_texts[i][j][k].set_position((posx, posy))
                cpu_bottom += cur_cpu_height
                mem_bottom += cur_mem_height
        
        self.suptitle.set_text(f'[Attempt #{attempt}] action: {action}')
