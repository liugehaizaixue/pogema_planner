import numpy as np
from collections import deque

# BFS搜索最短路径
def bfs(start, destination, obstacles):
    rows, cols = obstacles.shape
    queue = deque([start])
    visited = set([start])
    parents = {start: None}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    while queue:
        current = queue.popleft()
        if current == tuple(destination):
            break
        
        for direction in directions:
            next_row, next_col = current[0] + direction[0], current[1] + direction[1]
            if 0 <= next_row < rows and 0 <= next_col < cols:
                next_cell = (next_row, next_col)
                if next_cell not in visited and obstacles[next_row, next_col] != 1:
                    queue.append(next_cell)
                    visited.add(next_cell)
                    parents[next_cell] = current
                    
    # 追踪路径
    path = []
    step = tuple(destination)
    while step is not None:
        path.append(step)
        step = parents[step]
    
    return path[::-1]

def calculate_path(matrix_obstacle , matrix_destination):
    destination = np.argwhere(matrix_destination == 1)[0]
    start = (matrix_obstacle.shape[0]//2, matrix_obstacle.shape[1]//2)
    path_matrix = np.zeros(matrix_obstacle.shape, dtype=int)
    try:
        path = bfs(start, destination, matrix_obstacle)
    except:
        path = [] # 没找到路径
    for row, col in path:
        path_matrix[row, col] = 1
    return path_matrix

if __name__ == '__main__':


    # 定义矩阵
    matrix_obstacle = np.array([
        [-1, 1, 0, 1, 0], 
        [-1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0]
    ])

    matrix_destination = np.array([
        [0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ])
    
    import time
    start_time = time.time()
    for i in range(10000):
        path_matrix = calculate_path(matrix_obstacle, matrix_destination)
    print(f"Path calculation took {time.time() - start_time} seconds.")
    print(path_matrix)