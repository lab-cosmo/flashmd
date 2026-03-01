import torch


# Disable static fusion. Besides the fact that atomistic batches have variable
# sizes, statically fused CUDA kernels cannot allocate new tensors at runtime,
# causing "Global alloc not supported yet" errors (cuda 13+) at the time of writing
torch.jit.set_fusion_strategy([("DYNAMIC", 10)])
