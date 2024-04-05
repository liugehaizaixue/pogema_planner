import numpy as np
from pogema import GridConfig
import random
from planning.astar_no_grid import AStar, INF
from utils.utils_debug import data_visualizer
from copy import deepcopy

class RePlanBase:
    def __init__(self, use_best_move: bool = True, max_steps: int = INF, algo_source='c++', seed=None,
                 ignore_other_agents=False):

        self.use_best_move = use_best_move
        gc: GridConfig = GridConfig()

        # self.actions = {tuple(gc.MOVES[i]): i for i in range(len(gc.MOVES))}
        self.actions = {gc.NEW_MOVES[i]: i for i in range(len(gc.NEW_MOVES))}
        self.steps = 0


        self.algo_source = AStar
        self.planner = None
        self.max_steps = max_steps
        self.previous_positions = None
        self.rnd = np.random.default_rng(seed)
        self.ignore_other_agents = ignore_other_agents

    def act(self, obs, skip_agents=None):
        num_agents = len(obs)
        if self.planner is None:
            self.planner = [self.algo_source(self.max_steps) for _ in range(num_agents)]
        if self.previous_positions is None:
            self.previous_positions = [[] for _ in range(num_agents)]
        obs_radius = len(obs[0]['obstacles']) // 2
        action = []

        for k in range(num_agents):
            self.previous_positions[k].append([obs[k]['xy'],obs[k]['direction']])
            if obs[k]['xy'] == obs[k]['target_xy']:
                action.append(None)
                continue
            # obstacles = np.transpose(np.nonzero(obs[k]['obstacles']))
            obstacles = np.transpose(np.where(obs[k]['obstacles'] == 1))
            if self.ignore_other_agents:
                other_agents = None
            else:
                # other_agents = np.transpose(np.nonzero(obs[k]['agents']))
                other_agents = np.transpose(np.where(obs[k]['agents'] == 1))
            self.planner[k].update_obstacles(obstacles, other_agents,
                                             (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius))

            if skip_agents and skip_agents[k]:
                action.append(None)
                continue

            self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])
            # path = self.planner[k].get_next_node(self.use_best_move)
            path = self.planner[k].get_path(self.use_best_move)
            if data_visualizer.ego_idx is not None:
                if k == data_visualizer.ego_idx:
                    data_visualizer.ego_explored_map = self.planner[k].get_obstacles()
                    data_visualizer.ego_path = deepcopy(path)
            if path is not None and path[1][0] < INF:
                direction = obs[k]["direction"]
                action.append(self.actions[self.generate_action(path[0],path[1],direction)])
                # action.append(self.actions[(path[1][0] - path[0][0], path[1][1] - path[0][1])])
            else:
                action.append(None)
        self.steps += 1
        return action
    
    @staticmethod
    def generate_action(start,target,direction):
        """ 根据xy 与 target_xy发现
            x轴 向下为正， y轴向右为正 
            即 10,-24 位于 0,0的 左下方 下发10, 左方24
            通过x' = y ; y' = -x进行坐标转换
        """
        target_direction = [target[1] - start[1], -(target[0] - start[0])]
        # 计算两个向量的点积
        dot_product = direction[0] * target_direction[0] + direction[1] * target_direction[1]
        # 如果点积为正，说明两个向量同向
        if dot_product > 0:
            action = "FORWARD"
        # 如果点积为负，说明两个向量反向
        elif dot_product < 0:
            action = random.choice(["TURN_LEFT", "TURN_RIGHT"])
        else:
            # 否则，需要检查两个向量是否垂直
            # 如果两个向量垂直，它们没有左右关系，因此我们将返回 "垂直"
            if direction[0] * target_direction[1] - direction[1] * target_direction[0] == 0:
                """ target_direction为 0向量 """
                action = 'WAIT'
            # 如果两个向量不是垂直的，则它们必然有左右关系
            elif direction[0] * target_direction[1] - direction[1] * target_direction[0] > 0:
                action = "TURN_LEFT"
            else:
                action = "TURN_RIGHT"
        return action

    def get_path(self):
        results = []
        for idx in range(len(self.planner)):
            results.append(self.planner[idx].get_path(use_best_node=False))
        return results


class FixNonesWrapper:

    def __init__(self, agent):
        self.agent = agent
        self.rnd = self.agent.rnd
        # self.env = agent.env

    def act(self, obs, skip_agents=None):
        actions = self.agent.act(obs, skip_agents=skip_agents)
        for idx in range(len(actions)):
            if actions[idx] is None:
                actions[idx] = 0
        return actions


class NoPathSoRandomOrStayWrapper:

    def __init__(self, agent):
        self.agent = agent
        self.rnd = self.agent.rnd

    def act(self, obs, skip_agents=None):
        actions = self.agent.act(obs, skip_agents=skip_agents)
        for idx in range(len(actions)):
            if actions[idx] is None:
                if self.rnd.random() <= 0.5:
                    actions[idx] = 0
                else:
                    actions[idx] = self.get_random_move(obs, idx)
        return actions

    def get_random_move(self, obs, agent_id):
        actions = [0, 2, 3]

        self.agent.rnd.shuffle(actions) #打乱顺序
        direction = obs[agent_id]['direction']
        
        i = len(obs[agent_id]['obstacles']) // 2 + direction[0]
        j = len(obs[agent_id]['obstacles']) // 2 + direction[1]
        if obs[agent_id]['obstacles'][i][j] == 0:
                return 1
        return actions[0]


class FixLoopsWrapper(NoPathSoRandomOrStayWrapper):
    def __init__(self, agent, stay_if_loop_prob=None, add_none_if_loop=False):
        super().__init__(agent)
        self.rnd = self.agent.rnd
        self.previous_positions = None
        self.stay_if_loop_prob = stay_if_loop_prob if stay_if_loop_prob else 0.5
        self.add_none_if_loop = add_none_if_loop

    def act(self, obs, skip_agents=None):
        num_agents = len(obs)
        if self.previous_positions is None:
            self.previous_positions = [[] for _ in range(num_agents)]

        actions = self.agent.act(obs, skip_agents=skip_agents)
        for idx in range(len(actions)):
            if actions[idx] is None:
                continue
            path = self.previous_positions[idx]
            if len(path) > 1:
                next_step = obs[idx]['xy']
                previous_direction = obs[idx]['direction']
                action = GridConfig().NEW_MOVES[actions[idx]]
                """ 由于坐标系不同，x’ = y ； y'= -x 
                    direction[0] , direction[1]  = direction[1] , -direction[0]
                """
                if action == "FORWARD":
                    next_pos = next_step[0] + previous_direction[1] , next_step[1] -previous_direction[0]
                    next_direction = previous_direction
                elif action == "WAIT":
                    next_pos = next_step
                    next_direction = previous_direction
                else:
                    next_pos = next_step
                    if action == "TURN_LEFT":
                        next_direction = [-previous_direction[1], previous_direction[0]]
                    elif action == "TURN_RIGHT":
                        next_direction = [previous_direction[1], -previous_direction[0]]

                # if path[-1] == [next_pos, next_direction] or path[-2] == [next_pos, next_direction]:
                #     if self.add_none_if_loop:
                #         actions[idx] = None
                #     elif next_pos == next_step:
                #         actions[idx] = self.get_random_move(obs, idx)
                #     elif self.rnd.random() < self.stay_if_loop_prob:
                #         actions[idx] = 0
            self.previous_positions[idx].append([obs[idx]['xy'], obs[idx]['direction']])
        return actions
