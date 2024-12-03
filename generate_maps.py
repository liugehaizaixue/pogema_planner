import numpy as np
from dataclasses import dataclass

@dataclass
class MapRangeSettings:
    width_min: int = 20
    width_max: int = 20
    height_min: int = 20
    height_max: int = 20
    obstacle_density_min: float = 0.1
    obstacle_density_max: float = 0.1

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)
        return {
            "width": rng.integers(self.width_min, self.width_max + 1),
            "height": rng.integers(self.height_min, self.height_max + 1),
            "obstacle_density": rng.uniform(self.obstacle_density_min, self.obstacle_density_max),
            "seed": seed
        }

@dataclass
class WarehouseConfig:
    wall_width: int = 5
    wall_height: int = 2
    walls_in_row: int = 5
    walls_rows: int = 5
    bottom_gap: int = 5
    horizontal_gap: int = 2
    vertical_gap: int = 3


def generate_map(settings):
    rng = np.random.default_rng(settings["seed"])
    width, height, obstacle_density = settings["width"], settings["height"], settings["obstacle_density"]
    map_data = [['.' for _ in range(width)] for _ in range(height)]
    total_tiles = width * height
    total_obstacles = int(total_tiles * obstacle_density)

    obstacles_placed = 0
    while obstacles_placed < total_obstacles:
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        if map_data[y][x] == '.':
            map_data[y][x] = '#'
            obstacles_placed += 1

    return '\n'.join(''.join(row) for row in map_data)


def generate_warehouse(cfg: WarehouseConfig):
    height = cfg.vertical_gap * (cfg.walls_rows + 1) + cfg.wall_height * cfg.walls_rows
    width = cfg.bottom_gap * 2 + cfg.wall_width * cfg.walls_in_row + cfg.horizontal_gap * (cfg.walls_in_row - 1)

    grid = np.zeros((height, width), dtype=int)

    for row in range(cfg.walls_rows):
        row_start = cfg.vertical_gap * (row + 1) + cfg.wall_height * row
        for col in range(cfg.walls_in_row):
            col_start = cfg.bottom_gap + col * (cfg.wall_width + cfg.horizontal_gap)
            grid[row_start:row_start + cfg.wall_height, col_start:col_start + cfg.wall_width] = 1

    return '\n'.join(''.join('.' if cell == 0 else '#' for cell in row) for row in grid)

def warehouse_generate_and_save_maps(config=WarehouseConfig()):
    test_maps = {}
    map_data = generate_warehouse(config)
    map_name = f"warehouse-{config.wall_width}-{config.wall_height}"
    test_maps[map_name] = map_data

    maps_dict_to_yaml(f'warehouse.yaml', test_maps)


def random_generate_and_save_maps(density, seed_range=10):
    test_maps = {}
    settings_generator = MapRangeSettings()
    settings_generator.obstacle_density_max = density
    settings_generator.obstacle_density_min = density

    for seed in range(seed_range):
        settings = settings_generator.sample(seed)
        map_data = generate_map(settings)
        map_name = f"density{density}-seed-{str(seed)}"
        test_maps[map_name] = map_data

    maps_dict_to_yaml(f'random-density{density}.yaml', test_maps)


import yaml
import os
def maps_dict_to_yaml(filename, maps):
    folder = os.path.dirname("exp_maps/")
    if folder and not os.path.exists(folder):
        os.makedirs(folder)  # 创建文件夹

    with open("exp_maps/"+filename, 'w') as file:
        yaml.add_representer(str,
                             lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|'))
        yaml.dump(maps, file)


if __name__ == '__main__':
    random_generate_and_save_maps(0)
    random_generate_and_save_maps(0.1)
    random_generate_and_save_maps(0.2)
    random_generate_and_save_maps(0.3)
    warehouse_generate_and_save_maps()


