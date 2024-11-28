from agents.utils_agents import  run_algorithm
from planning.cbs import CBS_Planner
from planning.astar_for_cbs import AStarWithDirection
from pogema import GridConfig




class CBS:
    def __init__(self):
        self.agent = None
        self.env = None
        self.planner = None
        self.paths = []
        self.actions = []
        gc: GridConfig = GridConfig()
        self.actions_dict = {gc.NEW_MOVES[i]: i for i in range(len(gc.NEW_MOVES))}
        pass


    def act(self, observations, rewards=None, dones=None, info=None):
        # [TODO]: Directly return the next action of each intelligent agent
        actions = []
        for i in range(len(observations)):
            action = self.actions[i].pop(0) if self.actions[i] else self.actions_dict['WAIT']
            actions.append(action)
        return actions

    def after_step(self, dones):
        if all(dones):
            self.agent = None

    def after_reset(self, global_obstacles, Starts, Starts_directions, Goals):
        # [TODO]: cbs for all agents , record the paths
        self.paths.clear()
        self.actions.clear()
        self.planner = None
        self.planner = CBS_Planner(map=global_obstacles, Starts=Starts, Starts_directions=Starts_directions, Goals=Goals)
        res = self.planner.FindSolution()
        if res[0]:
            for agent, sol in res[1].sol.items():
                path = []
                for node in sol:
                    path.append((node.i, node.j, node.direction, node.t))
                self.actions.append(self.convert_to_actions(path))
                self.paths.append(path)
            print("Solution found")
        else:
            print("Solution not found")
            raise Exception("Solution not found")


    def convert_to_actions(self, path):
        action = []
        for i in range(len(path)-1):
            action.append(self.actions_dict[AStarWithDirection.generate_action(path[i], path[i+1])])
        return action

def example_cbs(map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64, animate=False):
    algo = CBS()
    return run_algorithm(algo, map_name, max_episode_steps, seed, num_agents, animate)
