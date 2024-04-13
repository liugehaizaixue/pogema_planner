import multiprocessing
from typing import Optional

from pomapf_env.pomapf_config import POMAPFConfig

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra, BaseModel, validator
import os
from os.path import join


class Environment(BaseModel, ):
    grid_config: POMAPFConfig = POMAPFConfig(observation_type="POMAPF")
    env: str = "POMAPF-v0"
    grid_memory_obs_radius: Optional[int] = 7 #None
    with_animation: bool = False
    worker_index: int = None
    vector_index: int = None
    env_id: int = None
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = [64, 128, 256, 256]
    use_maps: bool = True
    every_step_metrics: bool = False

class EncoderConfig(BaseModel):
    """
    Configuration for an encoder.
    Args:
        extra_fc_layers (int): Number of extra fully connected (fc) layers. Default is 1.
        num_filters (int): Number of filters. Default is 64.
        num_res_blocks (int): Number of residual blocks. Default is 1.
        nonlinearity (Literal['elu', 'relu', 'tanh']): Activation function to use. Default is 'relu'.
        hidden_size (int): Hidden size for extra fc layers. Default is 128.
    """
    extra_fc_layers: int = 1
    num_filters: int = 64
    num_res_blocks: int = 8
    nonlinearity: Literal['elu', 'relu', 'tanh'] = 'relu'
    hidden_size: int = 512

class Experiment(BaseModel):
    environment: Environment = Environment()
    encoder: EncoderConfig = EncoderConfig()

    env: Literal['POMAPF-v0'] = "POMAPF-v0"

    force_envs_single_thread: bool = True
    restart_behavior: str = "resume"  # ["resume", "restart", "overwrite"]
    normalize_returns: bool = False
    async_rl: bool = False
    num_batches_per_epoch: int = 16

    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers = []
    save_best_metric: str = "avg_throughput"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1

    keep_checkpoints: int = 1
    stats_avg: int = 100
    
    # ============================
    train_for_env_steps: int = 1e10
    train_for_seconds: int = 1e10

    # ============================
    learning_rate: float = 1e-4
    gamma: float = 0.99
    adam_eps: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    gae_lambda: float = 0.95
    rollout: int = 32
    recurrence: int = 32

    use_rnn: bool = True
    rnn_size: int = 256
    rnn_type: str = 'gru'
    rnn_num_layers: int = 1

    ppo_clip_ratio: float = 0.1
    ppo_clip_value: float = 1.0
    batch_size: int = 4096

    num_batches_per_epoch: int = 1
    num_epochs: int = 1

    max_grad_norm: float = 4.0
    optimizer: Literal["adam", "lamb"] = 'adam'

    exploration_loss_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    kl_loss_coeff: float = 0.0
    exploration_loss: str = 'entropy'

    max_policy_lag: int = 100
    # ============================
    with_pbt: bool = True
    pbt_optimize_gamma: bool = True
    pbt_mix_policies_in_one_env: bool = True
    pbt_period_env_steps: int = 5e6
    pbt_start_mutation: int = 2e7
    pbt_replace_fraction: float = 0.3
    pbt_mutation_rate: float = 0.15
    pbt_replace_reward_gap: float = 0.1
    pbt_replace_reward_gap_absolute: float = 1e-6
    pbt_target_objective: str = 'true_reward' #?

    # =============================
    num_workers: int = multiprocessing.cpu_count() # 4
    num_envs_per_worker: int = 4
    worker_num_splits: int = 1
    policy_initialization: str = 'orthogonal'
    # =============================

    lr_schedule: str = 'kl_adaptive_minibatch'
    lr_schedule_kl_threshold: float = None

    experiment: str = 'exp'
    train_dir: str = 'experiments/train_dir'
    seed: Optional[int] = 42
    use_wandb: bool = True


    save_every_sec: int = 120

    obs_subtract_mean: float = 0.0
    obs_scale: float = 1.0

    reward_scale: float = 1.0
    reward_clip: float = 10.0

    policy_init_gain: float = 1.0
    actor_critic_share_weights: bool = True

    adaptive_stddev: bool = True
    initial_stddev: float = 1.0