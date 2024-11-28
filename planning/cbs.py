from heapq import heappop, heappush
from planning.astar_for_cbs import AStarWithDirection
import copy
import time


# 表示底层的路径搜索节点
class LowNode:
    '''
    LowNode class represents a low-level search node

    - i, j: coordinates of corresponding grid element
    - direction: direction of corresponding grid element
    - t: time step of corresponding grid element
    '''

    def __init__(self, i, j, direction, t):
        self.t = t # 时间步
        self.i = i
        self.j = j
        self.direction = direction

def ReconstructPath(points):
    # 将points list 转为 path node list
    path = []
    for point in points:
        path.append(LowNode(point[0], point[1], point[2], point[3]))
    return path


# 上层节点
# CT树上的Node，用于CBS的CT树上的扩展
class HighNode:
    '''
    HighNode class represents a high-level search node

    - vertexCons: vertex constraints of the node
    - edgeCons: edge constraints of the node
    - sol: solution of the node
    - g: cost of the solution of the node 
    - h: h-value of the node
    - F: f-value of the node
    - parent: pointer to the parent-node 
    '''

    def __init__(self, vertexCons={}, edgeCons={}, sol={}, g=0, h=0, F=None, parent=None, k=0):
        self.vertexCons = vertexCons
        self.edgeCons = edgeCons
        self.sol = sol
        self.g = g
        self.h = h
        self.k = k
        if F is None:
            self.F = g + h
        else:
            self.F = F        
        self.parent = parent
    
    
    def __eq__(self, other):
        return (self.vertexCons == other.vertexCons) and (self.edgeCons == other.edgeCons) and \
               (self.sol == other.sol)
    
    def __lt__(self, other):
        return self.F < other.F or ((self.F == other.F) and (self.h < other.h)) \
        or ((self.F == other.F) and (self.h == other.h))

class OpenHigh:
    
    def __init__(self):
        self.heap = []
    
    def __iter__(self):
        return iter(self.heap)
    
    def __len__(self):
        return len(self.heap)

    def isEmpty(self):
        if len(self.heap) != 0:
            return False
        return True
    
    def AddNode(self, node : HighNode):
        heappush(self.heap, node)
        
    def GetBestNode(self):     
        best = heappop(self.heap)
        
        return best

class CBS_Planner:
    def __init__(self, map, Starts, Starts_directions, Goals, Time_limit = 300):
        self.map = map
        self.Starts = Starts
        self.Starts_directions = Starts_directions
        self.Goals = Goals
        self.OPEN = OpenHigh()
        self.agents = list(range(len(Starts)))
        self.time_limit = Time_limit

    # TODO: devide into two parts: vertex and edge constraint
    def checkConflict(self, s: HighNode):
        newVertexCons = []
        newEdgeCons = []
        # 每个agent的path都进行对比
        for i, a in enumerate(self.agents):
            for b in self.agents[i + 1 :]:
                for step in range(min(len(s.sol[a]), len(s.sol[b]))):
                    # 找到约束立即返回，不要把所有的Constraint都找到才结束
                    if s.sol[a][step].i == s.sol[b][step].i and s.sol[a][step].j == s.sol[b][step].j:
                        newVertexCons.append((a, b, (s.sol[a][step].i, s.sol[a][step].j), s.sol[a][step].t))
                        return newEdgeCons, newVertexCons
                    if step + 1 < min(len(s.sol[a]), len(s.sol[b])) and \
                       s.sol[a][step].i == s.sol[b][step + 1].i and s.sol[a][step].j == s.sol[b][step + 1].j and \
                       s.sol[a][step + 1].i == s.sol[b][step].i and s.sol[a][step + 1].j == s.sol[b][step].j:
                        newEdgeCons.append((
                            a,
                            b,
                            (s.sol[a][step].i, s.sol[a][step].j),
                            (s.sol[a][step + 1].i, s.sol[a][step + 1].j),
                            s.sol[a][step].t))
                        return newEdgeCons, newVertexCons
        # 没找到，返回本身此时都是空的list
        return newEdgeCons, newVertexCons

    def GenerateNodeSolution(self, node: HighNode):
        for a in self.agents:
            VC = []
            EC = []
            if a in node.vertexCons:
                VC = node.vertexCons[a]
            if a in node.edgeCons:
                EC = node.edgeCons[a]
            planner = AStarWithDirection()
            planner.update_obstacles(self.map)
            planner.update_time_constraints(VC)
            planner.update_time_constraints(EC)
            planner.update_path(self.Starts[a], self.Starts_directions[a], self.Goals[a])
            points = planner.get_path()
            if points:
                node.sol[a] = ReconstructPath(points)
            else:
                return False, None
        node.F = sum([len(path) for path in node.sol.values()])
        return True, node

    # TODO：修改为时间限制，因为可以无限制的进行冲突限制，时间够长一定有一个解。
    # 但是 内存限制，因此不能一直扩展下去
    def FindSolution(self):
        tic = time.perf_counter()
        generate = 0 # 记录生成的节点数量
        expand = 0 # 记录节点扩展的数量
        root = HighNode(vertexCons={}, edgeCons={}, sol={}, k=generate)
        res = self.GenerateNodeSolution(root)
        if res[0]:
            self.OPEN.AddNode(res[1])
        else:
            return (False, None, generate, expand)
        
        generate += 1
        toc = time.perf_counter()
        while toc - tic <= self.time_limit and not self.OPEN.isEmpty():
            node = self.OPEN.GetBestNode()

            expand += 1

            newEdgeCons, newVertexCons = self.checkConflict(node)

            # 路径没有新约束，因此该 solution 即为最终的解
            if len(newVertexCons) == 0 and len(newEdgeCons) == 0:
                return (True, node, generate, expand)
            
            # print('detect conflict')
            # 存在点约束
            if len(newVertexCons) > 0:
                a, b, (i, j), t = newVertexCons[0]

                # 左子结点
                vertex_cons_left = copy.deepcopy(node.vertexCons)
                edge_cons_left = copy.deepcopy(node.edgeCons)
                if a in vertex_cons_left:
                    vertex_cons_left[a].append(((i, j), t))
                else:
                    vertex_cons_left[a] = [((i, j), t)]
                left = HighNode(vertexCons=vertex_cons_left, edgeCons=edge_cons_left, sol={}, k=generate, parent=node)
                res = self.GenerateNodeSolution(left)

                if res[0]:
                    self.OPEN.AddNode(res[1])
                    generate+=1

                # 右子节点
                vertex_cons_right = copy.deepcopy(node.vertexCons)
                edge_cons_right = copy.deepcopy(node.edgeCons)
                if b in vertex_cons_right:
                    vertex_cons_right[b].append(((i, j), t))
                else:
                    vertex_cons_right[b] = [((i, j), t)]
                right = HighNode(vertexCons=vertex_cons_right, edgeCons=edge_cons_right, sol={}, k=generate, parent=node)
                res = self.GenerateNodeSolution(right)
                
                if res[0]:
                    self.OPEN.AddNode(res[1])
                    generate+=1
            else:
                a, b, (i1, j1), (i2, j2), t = newEdgeCons[0]
                # 左子结点
                vertex_cons_left = copy.deepcopy(node.vertexCons)
                edge_cons_left = copy.deepcopy(node.edgeCons)
                if a in edge_cons_left:
                    edge_cons_left[a].append(((i1, j1), (i2, j2), t))
                else:
                    edge_cons_left[a] = [((i1, j1), (i2, j2), t)]
                left = HighNode(vertexCons=vertex_cons_left, edgeCons=edge_cons_left, sol={}, k=generate, parent=node)
                res = self.GenerateNodeSolution(left)

                if res[0]:
                    self.OPEN.AddNode(res[1])
                    generate+=1

                # 右子节点
                vertex_cons_right = copy.deepcopy(node.vertexCons)
                edge_cons_right = copy.deepcopy(node.edgeCons)
                if b in edge_cons_right:
                    edge_cons_right[b].append(((i2, j2), (i1, j1), t))
                else:
                    edge_cons_right[b] = [((i2, j2), (i1, j1), t)]
                right = HighNode(vertexCons=vertex_cons_right, edgeCons=edge_cons_right, sol={}, k=generate, parent=node)
                res = self.GenerateNodeSolution(right)
                
                if res[0]:
                    self.OPEN.AddNode(res[1])
                    generate+=1
            toc = time.perf_counter()
        # open集合中不包含节点了，没有解
        return (False, None, generate, expand)

            
