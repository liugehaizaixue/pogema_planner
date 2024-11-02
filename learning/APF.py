import numpy as np

def normalize_matrices(matrix):
    abs_max = np.max(matrix)
    if abs_max == 0:
        return matrix
    return matrix / abs_max

def calculate_apf(obstacle_matrix, agents_matrix, destination_matrix, k_att=0.6, repulsion_coeff1=1, repulsion_coeff2=0.5, special_k_att=0.2):
    """ 计算人工势场（APF） """
    matrix = obstacle_matrix + agents_matrix
    destination = np.argwhere(destination_matrix == 1)[0]
    grid_x, grid_y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    dx = destination[1] - grid_x
    dy = destination[0] - grid_y
    dist_to_dest = np.square(dx) + np.square(dy)

    U_att = np.zeros_like(matrix, dtype=float)
    U_rep = np.zeros_like(matrix, dtype=float)
    U_special = np.zeros_like(matrix, dtype=float)
    # U_total = np.zeros_like(matrix, dtype=float)
    np.multiply(dist_to_dest, 0.5*k_att , out=U_att)


    for coeff, obstacles in [(repulsion_coeff1, obstacle_matrix == 1), (repulsion_coeff2, agents_matrix == 1)]:
        odx = grid_x[:, :, np.newaxis] - np.argwhere(obstacles)[:, 1]
        ody = grid_y[:, :, np.newaxis] - np.argwhere(obstacles)[:, 0]
        distance_to_obs = np.hypot(odx, ody)

        for i in range(distance_to_obs.shape[2]):
            dist = distance_to_obs[:, :, i]
            U_rep += 0.5 * coeff / dist**2  # 基于距离的平方反比计算

    # 添加-1点的处理
    special_obstacles = obstacle_matrix == -1
    sdx = grid_x[:, :, np.newaxis] - np.argwhere(special_obstacles)[:, 1]
    sdy = grid_y[:, :, np.newaxis] - np.argwhere(special_obstacles)[:, 0]
    special_magnitude = np.square(sdx) + np.square(sdy)
    for i in range(special_magnitude.shape[2]):
        dist = special_magnitude[:, :, i]
        U_special += 0.5 * special_k_att * dist

    # U_total = U_att + U_rep + U_special

    if np.any(np.isinf(U_rep)):
        max_value = 2*np.max(U_rep[~np.isinf(U_rep)])
        U_rep[np.isinf(U_rep)] = max_value

    return normalize_matrices(U_att) , normalize_matrices(U_rep) , U_special


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    # 定义5x5矩阵, -1 不可见 ， 0 可通行， 1静态障碍物，2智能体，3目的地
    matrix_obstacle = np.array([
        [-1, 1, 0, 1, 0], 
        [-1, 1, 0, 1, 0],
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
    for i in range(1000):
        U_att , U_rep , U_special = calculate_apf(matrix_obstacle, matrix_agents, matrix_destination)
    end_time = time.time()
    print(f"APF calculation took {end_time - start_time} seconds.")

    # U_total = U_att + U_special + U_rep
    U_total = U_att + U_rep

    # if np.any(np.isinf(U_total)):
    #     max_value = 1.2*np.max(U_total[~np.isinf(U_total)])
    #     U_total[np.isinf(U_total)] = max_value
    # print(U_total)
    # # 计算梯度
    grad_y, grad_x = np.gradient(U_total , edge_order=1)

    print(grad_x)

    print(grad_y)

    # 绘制势场梯度
    fig, ax = plt.subplots()
    ax.imshow(matrix_destination, cmap='gray', origin='upper')

    # 初始化网格，用于可视化势场向量
    x_vals, y_vals = np.meshgrid(np.arange(matrix_agents.shape[0]), np.arange(matrix_agents.shape[1]))
    
    # 绘制向量场
    ax.quiver(x_vals, y_vals, -grad_x, grad_y, color='r')

    # 标记目的地
    destination = np.argwhere(matrix_destination == 1)[0]
    ax.text(destination[1], destination[0], 'D', color='blue', ha='center', va='center', fontsize=16)

    plt.title("Matrix with Artificial Potential Field (APF)")

    plt.show()

    
    # 绘制吸引势能、斥力势能、特殊斥力势能和总势能
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    # 吸引势能图
    im1 = ax[0].imshow(U_att, cmap='viridis', origin='lower')
    ax[0].set_title("Attractive Potential")
    ax[0].set_xticks(range(matrix_obstacle.shape[1]))
    ax[0].set_yticks(range(matrix_obstacle.shape[0]))
    ax[0].set_xticklabels(range(1, matrix_obstacle.shape[1] + 1))
    ax[0].set_yticklabels(range(1, matrix_obstacle.shape[0] + 1))
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].invert_yaxis()  # 反转 y 轴

    # 斥力势能图
    im2 = ax[1].imshow(U_rep, cmap='viridis', origin='lower')
    ax[1].set_title("Repulsive Potential")
    ax[1].set_xticks(range(matrix_obstacle.shape[1]))
    ax[1].set_yticks(range(matrix_obstacle.shape[0]))
    ax[1].set_xticklabels(range(1, matrix_obstacle.shape[1] + 1))
    ax[1].set_yticklabels(range(1, matrix_obstacle.shape[0] + 1))
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].invert_yaxis()  # 反转 y 轴

    # 特殊斥力势能图
    im3 = ax[2].imshow(U_special, cmap='viridis', origin='lower')
    ax[2].set_title("Special Repulsive Potential")
    ax[2].set_xticks(range(matrix_obstacle.shape[1]))
    ax[2].set_yticks(range(matrix_obstacle.shape[0]))
    ax[2].set_xticklabels(range(1, matrix_obstacle.shape[1] + 1))
    ax[2].set_yticklabels(range(1, matrix_obstacle.shape[0] + 1))
    fig.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].invert_yaxis()  # 反转 y 轴

    # 总势能图
    im4 = ax[3].imshow(U_total, cmap='viridis', origin='lower')
    ax[3].set_title("Total Potential")
    ax[3].set_xticks(range(matrix_obstacle.shape[1]))
    ax[3].set_yticks(range(matrix_obstacle.shape[0]))
    ax[3].set_xticklabels(range(1, matrix_obstacle.shape[1] + 1))
    ax[3].set_yticklabels(range(1, matrix_obstacle.shape[0] + 1))
    fig.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].invert_yaxis()  # 反转 y 轴

    plt.tight_layout()
    plt.show()