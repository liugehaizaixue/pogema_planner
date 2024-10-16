import torch
import numpy as np

# 将矩阵移动到GPU
def to_device(matrix, device):
    return torch.tensor(matrix, dtype=torch.float32, device=device)

def normalize_matrices(matrix1, matrix2):
    min_value = torch.min(torch.min(matrix1), torch.min(matrix2))
    max_value = torch.max(torch.max(matrix1), torch.max(matrix2))

    abs_value = torch.max(torch.abs(max_value), torch.abs(min_value))

    def normalize(matrix, abs_value):
        if abs_value == 0:
            return matrix
        return matrix / abs_value

    normalized_matrix1 = normalize(matrix1, abs_value)
    normalized_matrix2 = normalize(matrix2, abs_value)
    return normalized_matrix1.cpu().numpy(), normalized_matrix2.cpu().numpy()

def calculate_apf(obstacle_matrix, agents_matrix, destination_matrix, repulsion_coeff1=0.3, repulsion_coeff2=0.3, device='cuda'):
    # 将矩阵移动到 GPU
    obstacle_matrix = to_device(obstacle_matrix, device)
    agents_matrix = to_device(agents_matrix, device)
    destination_matrix = to_device(destination_matrix, device)

    matrix = obstacle_matrix + agents_matrix
    destination = torch.nonzero(destination_matrix == 1, as_tuple=False)[0]

    # 注意这里移除了 `indexing` 参数，默认行为是 'xy'
    grid_y, grid_x = torch.meshgrid(torch.arange(matrix.shape[0], device=device, dtype=torch.float32), 
                                    torch.arange(matrix.shape[1], device=device, dtype=torch.float32))

    dx = destination[1] - grid_x
    dy = destination[0] - grid_y
    dist_to_dest = torch.sqrt(dx**2 + dy**2)

    attraction_x = torch.where(dist_to_dest != 0, dx / dist_to_dest, torch.zeros_like(dx))
    attraction_y = torch.where(dist_to_dest != 0, -dy / dist_to_dest, torch.zeros_like(dy))

    potential_vectors_x = attraction_x.clone()
    potential_vectors_y = attraction_y.clone()

    obstacles_type1 = torch.nonzero(obstacle_matrix == 1, as_tuple=False)
    obstacles_type2 = torch.nonzero(agents_matrix == 1, as_tuple=False)

    for obs in obstacles_type1:
        odx = grid_x - obs[1]
        ody = grid_y - obs[0]
        obs_magnitude = torch.sqrt(odx**2 + ody**2)
        repulsion_x = torch.where(obs_magnitude != 0, (odx / obs_magnitude**2) * repulsion_coeff1, torch.zeros_like(odx))
        repulsion_y = torch.where(obs_magnitude != 0, (-ody / obs_magnitude**2) * repulsion_coeff1, torch.zeros_like(ody))
        potential_vectors_x += repulsion_x
        potential_vectors_y += repulsion_y

    for obs in obstacles_type2:
        odx = grid_x - obs[1]
        ody = grid_y - obs[0]
        obs_magnitude = torch.sqrt(odx**2 + ody**2)
        repulsion_x = torch.where(obs_magnitude != 0, (odx / obs_magnitude**2) * repulsion_coeff2, torch.zeros_like(odx))
        repulsion_y = torch.where(obs_magnitude != 0, (-ody / obs_magnitude**2) * repulsion_coeff2, torch.zeros_like(ody))
        potential_vectors_x += repulsion_x
        potential_vectors_y += repulsion_y

    potential_vectors_x[matrix != 0] = 0
    potential_vectors_y[matrix != 0] = 0
    potential_vectors_x[destination_matrix != 0] = 0
    potential_vectors_y[destination_matrix != 0] = 0

    return normalize_matrices(potential_vectors_x, potential_vectors_y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 初始化输入数据
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

    # 使用 GPU 计算人工势场
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    potential_vectors_x, potential_vectors_y = calculate_apf(matrix_obstacle, matrix_agents, matrix_destination, device=device)
    # print(potential_vectors_x)
    # print(potential_vectors_y)

    # 将结果移回 CPU 进行绘图
    # 绘制结果
    fig, ax = plt.subplots()
    ax.imshow(matrix_destination, cmap='gray', origin='upper')

    x_vals, y_vals = np.meshgrid(np.arange(matrix_agents.shape[0]), np.arange(matrix_agents.shape[1]))
    ax.quiver(x_vals, y_vals, potential_vectors_x, potential_vectors_y, color='r')

    destination = np.argwhere(matrix_destination == 1)[0]
    ax.text(destination[1], destination[0], 'D', color='blue', ha='center', va='center', fontsize=16)

    plt.title("Matrix with Artificial Potential Field (APF) using GPU")
    plt.show()
