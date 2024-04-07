import numpy as np
from pogema import GridConfig
from planning.astar_no_grid import AStar
from planning.astar_no_grid_with_direction import AStarWithDirection, INF
# from utils.utils_debug import data_visualizer
from copy import deepcopy

class RePlanBase:
    def __init__(self, use_best_move: bool = True, max_steps: int = INF, algo_name='A-with-direction', seed=None,
                 ignore_other_agents=False):

        self.use_best_move = use_best_move
        gc: GridConfig = GridConfig()

        # self.actions = {tuple(gc.MOVES[i]): i for i in range(len(gc.MOVES))}
        self.actions = {gc.NEW_MOVES[i]: i for i in range(len(gc.NEW_MOVES))}
        self.steps = 0
        self.algo_name = algo_name
        if algo_name == 'A-with-direction':
            self.algo_source = AStarWithDirection
        else:
            self.algo_source = AStar
        self.planner = None
        self.max_steps = max_steps
        self.rnd = np.random.default_rng(seed)
        self.ignore_other_agents = ignore_other_agents

    def act(self, obs, skip_agents=None):
        num_agents = len(obs)
        if self.planner is None:
            self.planner = [self.algo_source(self.max_steps) for _ in range(num_agents)]
        obs_radius = len(obs[0]['obstacles']) // 2
        action = []

        for k in range(num_agents):
            if obs[k]['xy'] == obs[k]['target_xy']:
                action.append(None)
                continue
            # obstacles = np.transpose(np.nonzero(obs[k]['obstacles']))
            obstacles = np.transpose(np.where(obs[k]['obstacles'] == 1))
            if self.ignore_other_agents:
                other_agents = None
            else:
                # other_agents = np.transpose(np.nonzero(obs[k]['agents']))
                if self.algo_name == 'A-with-direction':
                # 忽略自身，以实现带方向的A*，即可以停留在原地
                    _agents_matrix = deepcopy(obs[k]['agents'])
                    _agents_matrix[obs_radius][obs_radius] = 0
                    other_agents = np.transpose(np.where(_agents_matrix == 1))
                else:
                    other_agents = np.transpose(np.where(obs[k]['agents'] == 1))
            self.planner[k].update_obstacles(obstacles, other_agents,
                                             (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius))

            if skip_agents and skip_agents[k]:
                action.append(None)
                continue
            
            self.planner[k].update_path(obs[k]['xy'],obs[k]["direction"], obs[k]['target_xy'])
            # path = self.planner[k].get_next_node(self.use_best_move)
            path = self.planner[k].get_path(self.use_best_move)
            # if data_visualizer.ego_idx is not None:
            #     if k == data_visualizer.ego_idx:
            #         data_visualizer.ego_explored_map = self.planner[k].get_obstacles()
            #         data_visualizer.ego_path = deepcopy(path)
            if path is not None and path[1][0] < INF:
                direction = obs[k]["direction"]
                action.append(self.actions[self.planner[k].generate_action(path[0],path[1],direction)])
            else:
                action.append(None)
        self.steps += 1
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
            current_position = (obs[idx]['xy'] , obs[idx]['direction'])
            if len(path) > 1:
                c_xy = obs[idx]['xy']
                c_direction = obs[idx]['direction']
                action = GridConfig().NEW_MOVES[actions[idx]]
                """ 由于坐标系不同，x’ = -y ； y'= x 
                    direction[0] , direction[1]  = -direction[1] , direction[0]
                """
                if action == "FORWARD":
                    n_xy = (c_xy[0] - c_direction[1] , c_xy[1] + c_direction[0] )
                    n_direction = c_direction
                    next_position = ( n_xy , n_direction )
                elif action == "WAIT":
                    next_position = ( c_xy , c_direction )
                else:
                    if action == "TURN_LEFT":
                        n_direction = [-c_direction[1], c_direction[0]]
                        next_position = (c_xy, n_direction)
                    elif action == "TURN_RIGHT":
                        n_direction = [c_direction[1], -c_direction[0]]
                        next_position = (c_xy, n_direction)

                last_few_path = path[-6:] # 从之前的六个时间步来判断是否存在loop
                if next_position in last_few_path:
                    if self.add_none_if_loop:
                        actions[idx] = None
                    # elif next_pos == next_step:
                    #     actions[idx] = self.get_random_move(obs, idx)
                    elif self.rnd.random() < self.stay_if_loop_prob: # 0.5概率保留原动作/wait
                        actions[idx] = 0
            self.previous_positions[idx].append( current_position )
        return actions
