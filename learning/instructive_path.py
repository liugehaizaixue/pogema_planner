
import gymnasium
import numpy as np
from gymnasium.spaces import Box
from pogema import GridConfig
from pogema.grid import Grid
from copy import deepcopy
from heapq import heappop, heappush
import numpy as np
INF = 1000000000


class Node:
    def __init__(self, coord: (int, int, list) = (INF, INF, [1,0]) ,g: int = 0, h: int = 0):
        self.i, self.j , self.z= coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or \
               (self.f == other.f and (self.g < other.g or
                                       (self.g == other.g and (self.i < other.i or
                                                               (self.i == other.i and self.j < other.j)))))

class AstarPlanner:
    def __init__(self, max_steps: int = INF):
        self.start = None
        self.goal = None
        self.last_pos = None
        self.max_steps = max_steps
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()
        self.best_node = None
        self.desired_position = None
        self.bad_actions = set()

    @staticmethod
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return angle_deg        

    def h(self, node: [int, int, list]):
        """ 曼哈顿距离 加上 方向距离 正对方向 -1 , 反方向 +1  垂直方向0"""
        h0 = abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])
        v1 = [self.goal[0] - node[0] , self.goal[1] - node[1]]
        direction = node[2]
        v2 = [ -direction[1] , direction[0] ]
        if all(element == 0 for element in v1):
            #v1 为0向量，即已经到达目标点，此时h1应该为-1
            h1 = -1
        else:
            angle_deg = self.angle_between_vectors( v1 , v2)
            if angle_deg < 90:
                h1 = -1
            elif angle_deg == 90:
                h1 = 0
            else:
                h1 = 1
        return h0 + h1   

    def get_neighbours(self, u: [int, int, list]):
        neighbors = []
        candidates = []
        direction = u[2]
        candidates.append((u[0], u[1], (-direction[1], direction[0]) ) ) # TURN_LEFT
        candidates.append((u[0], u[1], (direction[1], -direction[0]) ) ) # TURN_RIGHT
        candidates.append((u[0]-direction[1] , u[1]+direction[0] , (direction[0], direction[1]) ) ) # FORWARD

        for c in candidates:
            if (c[0], c[1]) not in self.obstacles:
                neighbors.append(c)
        return neighbors
    def compute_shortest_path(self):
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            if self.best_node.h > u.h:
                self.best_node = u
            steps += 1
            for n in self.get_neighbours([u.i, u.j, u.z]):
                if n not in self.CLOSED and n not in self.other_agents:
                    heappush(self.OPEN, Node( n , u.g + 1, self.h(n)))
                    self.CLOSED[n] = (u.i, u.j, u.z)

    def update_obstacles(self, obs, other_agents, n):
        for o in obs:
            self.obstacles.add((n[0] + o[0], n[1] + o[1]))
        self.other_agents.clear()
        if other_agents:
            for a in other_agents:
                self.other_agents.add((n[0] + a[0], n[1] + a[1]))
    def reset(self):
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start, 0, self.h(self.start)))
        self.best_node = Node(self.start, 0, self.h(self.start))

    def update_path(self, start, start_direction, goal):
        self.start = (start[0], start[1], (start_direction[0], start_direction[1]))
        self.goal = goal
        self.reset()
        self.compute_shortest_path()

    def get_path(self, use_best_node: bool):
        path = []
        next_node = None
        goal = self.find_goal_in_closed()
        if goal:
            next_node = goal
        elif use_best_node:
            next_node = (self.best_node.i, self.best_node.j, self.best_node.z)
        if next_node is not None and next_node != self.start:
            while self.CLOSED[next_node] != self.start:
                path.append(next_node)
                next_node = self.CLOSED[next_node]
            path.append(next_node)
            path.append(self.start)
            self.desired_position = next_node
            path.reverse()
            return path
        else:
            self.desired_position = None
            return None
     
    def find_goal_in_closed(self):
        t_x , t_y  =self.goal
        result = [(x, y, z) for (x, y, z) in self.CLOSED if x == t_x and y == t_y]
        if result:
            return result[0]
        return None


class MultiplePlanner:
    def __init__(self, obs_radius):
        self.planners = None
        self.obs_radius = obs_radius
        self.ignore_other_agents = True
        self.use_best_move = True
        self.paths = []
    
    def update(self, observations):
        if self.planners is None or len(self.planners) != len(observations):
            self.planners = [AstarPlanner() for _ in range(len(observations))]
            self.obs_shape = (self.obs_radius * 2 + 1 , self.obs_radius * 2 + 1 )
        for agent_idx, obs in enumerate(observations):
            if obs['xy'] == obs['target_xy']:
                self.paths.append( np.zeros(self.obs_shape) )
                continue
            obstacles = np.transpose(np.where(obs['obstacles'] == 1))
            if self.ignore_other_agents:
                other_agents = None
            else:
                _agents_matrix = deepcopy(obs['agents'])
                _agents_matrix[self.obs_radius][self.obs_radius] = 0
                other_agents = np.transpose(np.where(_agents_matrix == 1))

            self.planners[agent_idx].update_obstacles(obstacles, other_agents,
                                             (obs['xy'][0] - self.obs_radius, obs['xy'][1] - self.obs_radius))

            
            self.planners[agent_idx].update_path(obs['xy'],obs["direction"], obs['target_xy'])
            path = self.planners[agent_idx].get_path(self.use_best_move)
            path_point = set([(x,y) for x , y ,z in path])
            center = (obs['xy'][0] , obs['xy'][1])
            matrix = self.map_points_to_matrix(path_point, center)
            self.paths.append(matrix)

    def modify_observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            obs['instructive_path'] = self.paths[agent_idx]

    def map_points_to_matrix(self , point_list, center):
        matrix_size = self.obs_shape
        center_x, center_y = center
        matrix = np.zeros(matrix_size)
        for point in point_list:
            x, y = point
            # 进行坐标转换，相对于中心点的偏移量
            x_shifted = (x - center_x) + matrix_size[0] // 2
            y_shifted = (y - center_y) + matrix_size[1] // 2
            if 0 <= x_shifted < matrix_size[0] and 0 <= y_shifted < matrix_size[1]:
                matrix[x_shifted][y_shifted] = 1
        return matrix

    def clear(self):
        self.planners = None

class InstructivePath(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_radius):
        super().__init__(env)
        self.obs_radius = obs_radius
        size = self.obs_radius * 2 + 1
        self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
            obstacles=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
            agents=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
            instructive_path=gymnasium.spaces.Box(0.0, 1.0, shape=(size, size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            direction = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=int),
        )

        self.mp = MultiplePlanner(obs_radius)

    def observation(self, observations):
        self.mp.update(observations)
        self.mp.modify_observation(observations, self.obs_radius)
        return observations

    def reset(self,seed=None, **kwargs):
        self.mp.clear()
        obs , infos = self.env.reset(seed=None, **kwargs)
        return self.observation(obs) , infos
