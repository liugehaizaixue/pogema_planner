from heapq import heappop, heappush
import numpy as np
INF = 1000000000


class Node:
    def __init__(self, coord: (int, int, list) = (INF, INF, [1,0]) ,g: int = 0, h: int = 0, t: int = 0):
        self.i, self.j , self.z= coord
        self.g = g
        self.h = h
        self.t = t
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or \
               (self.f == other.f and (self.g < other.g or 
                                        (self.g == other.g and (self.i < other.i or
                                                                (self.i == other.i and self.j < other.j)))))


class AStarWithDirection:
    def __init__(self, max_steps: int = INF):
        self.start = None
        self.goal = None
        self.time_constraints = {}  # 时间约束，例如 {(x, y, t): True} 表示在时间t不能处于坐标(x, y)
        self.max_steps = max_steps
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
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

    def get_neighbours(self, u: [int, int, list, int]):
        neighbors = []
        candidates = []
        direction = u[2]
        candidates.append((u[0], u[1], (-direction[1], direction[0]), u[3]+1 ) ) # TURN_LEFT
        candidates.append((u[0], u[1], (direction[1], -direction[0]), u[3]+1 ) ) # TURN_RIGHT
        candidates.append((u[0]-direction[1] , u[1]+direction[0] , (direction[0], direction[1]), u[3]+1 ) ) # FORWARD

        for c in candidates:
            # x, y, direction, time
            if (c[0], c[1]) not in self.obstacles and (c[0], c[1], c[3]) not in self.time_constraints:
                neighbors.append(c)
        return neighbors

    def compute_shortest_path(self):
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for n in self.get_neighbours([u.i, u.j, u.z, u.t]):
                if n not in self.CLOSED:
                    heappush(self.OPEN, Node(coord=(n[0], n[1], n[2]), g=u.g + 1, h=self.h(n), t=n[3]))
                    self.CLOSED[n] = (u.i, u.j, u.z, u.t)

    def get_next_node(self):
        next_node = None
        goal = self.find_goal_in_closed()
        if goal:
            next_node = goal
        if next_node is not None and next_node != self.start:
            while self.CLOSED[next_node] != self.start:
                next_node = self.CLOSED[next_node]
            self.desired_position = next_node
            return self.start, next_node
        else:
            self.desired_position = None
            return None

    def update_obstacles(self, obs):
        m , n = obs.shape
        for i in range(m):
            for j in range(n):
                if obs[i, j] == 1:
                    o = (i, j)
                    self.obstacles.add((o[0], o[1]))
        pass
    
    def update_time_constraints(self, time_constraints):
        # The format is as follows 
        # [((18, 12), 17)]
        for tc in time_constraints:
            self.time_constraints[(tc[0][0], tc[0][1], tc[1])] = True

    def reset(self):
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start, 0, self.h(self.start), t=0 ))

    def update_path(self, start, start_direction, goal):
        self.start = (start[0], start[1], (start_direction[0], start_direction[1]))
        self.goal = goal
        self.reset()
        self.compute_shortest_path()


    def get_path(self):
        path = []
        next_node = None
        goal = self.find_goal_in_closed()
        if goal:
            next_node = goal
        if next_node is not None and next_node != self.start:
            while self.CLOSED[next_node] != (self.start[0], self.start[1], self.start[2], 0):
                path.append(next_node)
                next_node = self.CLOSED[next_node]
            path.append(next_node)
            path.append((self.start[0], self.start[1], self.start[2], 0))
            self.desired_position = next_node
            path.reverse()
            return path
        else:
            self.desired_position = None
            return None
    
    def get_obstacles(self):
        return self.obstacles
    
    def find_goal_in_closed(self):
        t_x , t_y  =self.goal
        result = [(x, y, z, t ) for (x, y, z, t ) in self.CLOSED if x == t_x and y == t_y]
        if result:
            return result[0]
        return None

    @staticmethod
    def generate_action(start,target,direction=None):
        """ 根据xy 与 target_xy发现
            x轴 向下为正， y轴向右为正 
            即 10,-24 位于 0,0的 左下方 下发10, 左方24
            通过x' = y ; y' = -x进行坐标转换
        """
        direction = start[2]
        if direction == target[2]:
            # 方向相同，则验证是否为前进
            if target[0] == start[0]-direction[1] and target[1] == start[1] + direction[0] :
                action =  "FORWARD"
            else:
                raise ValueError("invalid path planning")
        
        else:
            # 方向不同，则验证是否为转弯
            if start[0] != target[0] or start[1] != target[1]:
                #方向不同 且 位置不同，错误操作
                raise ValueError("invalid path planning")
            else:
                # 方向不同，但位置相同，即转弯
                if target[2] == (-direction[1], direction[0]) :
                    action =  "TURN_LEFT"
                elif target[2] == (direction[1], -direction[0]):
                    action =  "TURN_RIGHT"
                else:
                    raise ValueError("invalid path planning")
        
        return action