import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy

class DataVisualizer:
    def __init__(self, ego_idx = None):
        self.ego_idx = ego_idx
        self.reset()
 
    def show(self):
        self.show_ego_obs_obstacles()
        self.show_ego_obs_agents()
        self.show_ego_direction_and_action()
        self.show_ego_path()
        self.show_ego_explored_map()
        plt.tight_layout()
        plt.show()
        print("show ")
    
    def show_ego_obs_obstacles(self):
        plt.subplot(3, 2, 3)
        colors = ['gray', 'white', 'black']
        cmap = ListedColormap(colors)
        plt.imshow(self.ego_obs_obstacles, cmap=cmap, interpolation='nearest')
        plt.title('ego_obs_obstacles')

    def show_ego_obs_agents(self):
        plt.subplot(3, 2, 4)
        colors = ['gray', 'white', 'blue']
        cmap = ListedColormap(colors)
        plt.imshow(self.ego_obs_agents, cmap=cmap, interpolation='nearest')
        plt.title('ego_obs_agents')

    def show_ego_direction_and_action(self):
        plt.subplot(3, 2, 1)
        # 添加备注信息
        text = f"""action: {str(self.ego_action)}
direction: {str(self.ego_direction)}
"""
        plt.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=12)
        # 关闭坐标轴
        plt.axis('off')

    def show_ego_path(self):
        plt.subplot(3, 2, 6)
        max_x = max(abs(point[1]) for point in self.ego_path)
        max_y = max(abs(point[0]) for point in self.ego_path)
        # 计算栅格图的大小
        r = max(max_x,max_y)
        grid_height = r*2 + 1
        grid_width = r*2 + 1

        # 创建虚拟的栅格图
        grid = [[0] * grid_width for _ in range(grid_height)]
        # 提取路径的 x 坐标和 y 坐标
        for point in self.ego_path:
            x, y = point
            grid[-x+r][y+r] = 1

        plt.imshow(grid, cmap='gray', interpolation='none', origin='lower')
        plt.title('Path Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        # 显示图形
        # plt.grid(True)

    def show_ego_explored_map(self):
        plt.subplot(3, 2, 5)
        max_x = max(abs(point[1]) for point in self.ego_explored_map)
        max_y = max(abs(point[0]) for point in self.ego_explored_map)
        r = max(max_x,max_y)
        grid_height = r*2 + 1
        grid_width = r*2 + 1

        # 创建虚拟的栅格图
        grid = [[0] * grid_width for _ in range(grid_height)]
        # 提取路径的 x 坐标和 y 坐标
        for point in self.ego_explored_map:
            x, y = point
            grid[-x+r][y+r] = 1
        colors = ['white', 'black']
        cmap = ListedColormap(colors)
        plt.imshow(grid, cmap=cmap, interpolation='none', origin='lower')
        plt.title('ego_explored_map')
        plt.xlabel('X')
        plt.ylabel('Y')

    def reset(self):
        self.ego_obs_obstacles = []
        self.ego_obs_agents = []
        self.ego_path = []
        self.ego_explored_map = []
        self.ego_direction = []
        self.ego_action = ""

data_visualizer = DataVisualizer(ego_idx=0)