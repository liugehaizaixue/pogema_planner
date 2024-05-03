# import torch
# from sample_factory.model.encoder import Encoder, ResBlock
# from sample_factory.model.model_utils import  nonlinearity
# from sample_factory.algo.utils.torch_utils import calc_num_elements
# from sample_factory.utils.utils import log
# from torch import nn as nn
# from typing import Tuple

# class CustomDecoder(Decoder):
#     def __init__(self, cfg, decoder_input_size: Tuple[int, int]):
#         super().__init__(cfg)
#         # build custom decoder architecture
#         ...

#     def forward(self, core_output):
#         # custom forward logic
#         ...