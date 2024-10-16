import numpy as np

def normalize_matrices(matrix1, matrix2):
    abs_max = np.max(np.abs([matrix1, matrix2]))
    if abs_max == 0:
        return matrix1, matrix2
    return matrix1 / abs_max, matrix2 / abs_max

def calculate_apf(obstacle_matrix, agents_matrix, destination_matrix, repulsion_coeff1=0.3, repulsion_coeff2=0.3):
    """ 计算人工势场（APF） """
    matrix = obstacle_matrix + agents_matrix
    destination = np.argwhere(destination_matrix == 1)[0]
    grid_x, grid_y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    dx = destination[1] - grid_x
    dy = destination[0] - grid_y
    dist_to_dest = np.hypot(dx, dy)

    attraction_x = np.zeros_like(dx, dtype=float)  # 创建与 dx 同形状的零数组
    attraction_y = np.zeros_like(dy, dtype=float) # 创建与 dy 同形状的零数组

    # 安全地执行条件除法
    np.divide(dx, dist_to_dest, out=attraction_x, where=(dist_to_dest != 0))
    np.divide(-dy, dist_to_dest, out=attraction_y, where=(dist_to_dest != 0))


    potential_vectors_x = attraction_x
    potential_vectors_y = attraction_y

    # 使用单个调用来处理所有障碍物的斥力
    for coeff, obstacles in [(repulsion_coeff1, obstacle_matrix == 1), (repulsion_coeff2, agents_matrix == 1)]:
        odx = grid_x[:, :, np.newaxis] - np.argwhere(obstacles)[:, 1]
        ody = grid_y[:, :, np.newaxis] - np.argwhere(obstacles)[:, 0]
        obs_magnitude = np.hypot(odx, ody)
        repulsion_x = odx / np.maximum(obs_magnitude**2, 1e-10) * coeff
        repulsion_y = -ody / np.maximum(obs_magnitude**2, 1e-10) * coeff
        potential_vectors_x += np.sum(repulsion_x, axis=2)
        potential_vectors_y += np.sum(repulsion_y, axis=2)

    # 障碍物区域的势场为零
    zero_areas = (matrix != 0) | (destination_matrix != 0)
    potential_vectors_x[zero_areas] = 0
    potential_vectors_y[zero_areas] = 0

    return normalize_matrices(potential_vectors_x, potential_vectors_y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    # 定义5x5矩阵, -1 不可见 ， 0 可通行， 1静态障碍物，2智能体，3目的地
    matrix_obstacle = np.array([
        [0, 1, 0, 1, 0], 
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0]
    ])

    matrix_agents = np.array([
        [0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])

    matrix_destination = np.array([
        [0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ])


    # 计算人工势场
    start_time = time.time()
    for i in range(100):
        potential_vectors_x, potential_vectors_y = calculate_apf(matrix_obstacle, matrix_agents, matrix_destination)
    potential_vectors_x, potential_vectors_y = calculate_apf(matrix_obstacle, matrix_agents, matrix_destination)
    end_time = time.time()
    print(f"APF calculation took {end_time - start_time} seconds.")
    
    # 绘制矩阵和势场
    fig, ax = plt.subplots()
    ax.imshow(matrix_destination, cmap='gray', origin='upper')

    # 初始化网格，用于可视化势场向量
    x_vals, y_vals = np.meshgrid(np.arange(matrix_agents.shape[0]), np.arange(matrix_agents.shape[1]))
    
    # 绘制向量场
    ax.quiver(x_vals, y_vals, potential_vectors_x, potential_vectors_y, color='r')

    # 标记目的地
    destination = np.argwhere(matrix_destination == 1)[0]
    ax.text(destination[1], destination[0], 'D', color='blue', ha='center', va='center', fontsize=16)

    plt.title("Matrix with Artificial Potential Field (APF)")
    plt.show()
