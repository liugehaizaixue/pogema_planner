import numpy as np

def calculate_apf(obstacle_matrix, agents_matrix, destination_matrix, repulsion_coeff1=0.3, repulsion_coeff2=0.3):
    """ 计算人工势场（APF） """
    matrix = obstacle_matrix + agents_matrix
    # 目的地位置
    destination = np.argwhere(destination_matrix == 1)[0]

    # 障碍物位置
    obstacles_type1 = np.argwhere(obstacle_matrix == 1)  # 类型1障碍物
    obstacles_type2 = np.argwhere(agents_matrix == 1)  # 类型2障碍物

    # 计算向量场（人工势场）
    potential_vectors_x = np.zeros_like(matrix, dtype=float)
    potential_vectors_y = np.zeros_like(matrix, dtype=float)

    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i, j] == 0:
                # 向目的地的向量
                dx = destination[1] - j
                dy = destination[0] - i
                magnitude = np.sqrt(dx**2 + dy**2)
                attraction_x = dx / magnitude
                attraction_y = -dy / magnitude

                # 计算障碍物斥力
                repulsion_x, repulsion_y = 0, 0

                for obs in obstacles_type1:
                    odx = j - obs[1]
                    ody = i - obs[0]
                    obs_magnitude = np.sqrt(odx**2 + ody**2)
                    if obs_magnitude > 0:  # 避免除以零
                        repulsion_x += (odx / obs_magnitude**2) * repulsion_coeff1
                        repulsion_y += - (ody / obs_magnitude**2) * repulsion_coeff1

                for obs in obstacles_type2:
                    odx = j - obs[1]
                    ody = i - obs[0]
                    obs_magnitude = np.sqrt(odx**2 + ody**2)
                    if obs_magnitude > 0:  # 避免除以零
                        repulsion_x += (odx / obs_magnitude**2) * repulsion_coeff2
                        repulsion_y += - (ody / obs_magnitude**2) * repulsion_coeff2
                
                # 合成总向量
                potential_vectors_x[i, j] = attraction_x + repulsion_x
                potential_vectors_y[i, j] = attraction_y + repulsion_y
            else:
                # 障碍物不设向量
                potential_vectors_x[i, j] = 0
                potential_vectors_y[i, j] = 0

    return potential_vectors_x, potential_vectors_y

if __name__ == "__main__":
    import matplotlib.pyplot as plt
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
    potential_vectors_x, potential_vectors_y = calculate_apf(matrix_obstacle , matrix_agents, matrix_destination)

    print(potential_vectors_x[2,4], potential_vectors_y[2,4])
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
