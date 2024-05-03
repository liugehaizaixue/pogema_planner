# from sample_factory.model.encoder import Encoder
# from sample_factory.model.decoder import Decoder
# from sample_factory.model.core import ModelCore
# from sample_factory.model.actor_critic import ActorCritic
# from sample_factory.algo.utils.context import global_model_factory
# from custom_model.custom_core import CustomCore
# from custom_model.custom_encoder import CustomEncoder
# from custom_model.custom_decoder import CustomDecoder
# from torch import nn
# import torch

# class CustomActorCritic(ActorCritic):
#     def __init__(
#         self,
#         model_factory,
#         obs_space,
#         action_space,
#         cfg,
#     ):
#         super().__init__(obs_space, action_space, cfg)

#         self.encoder = CustomEncoder(cfg, obs_space)
#         self.core = CustomCore(cfg, self.encoder.get_out_size())
#         self.decoder = CustomDecoder(cfg, self.core.get_out_size())
#         self.critic_linear = nn.Linear(self.decoder.get_out_size())
#         self.action_parameterization = self.get_action_parameterization(
#             self.decoder.get_out_size()
#         ) 

#     def forward(self, normalized_obs_dict, rnn_states, values_only=False):
#         # forward logic
#         pass