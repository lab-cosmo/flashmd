import warnings

import torch

from .models import get_pretrained as get_pretrained
from .models import save_checkpoint as save_checkpoint


warnings.filterwarnings("ignore", category=UserWarning, message="custom data")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="`torch.jit.script`"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="`torch.jit.load`"
)


# Disable static fusion. Besides the fact that atomistic batches have variable
# sizes, statically fused CUDA kernels cannot allocate new tensors at runtime,
# causing "Global alloc not supported yet" errors (cuda 13+) at the time of writing
torch.jit.set_fusion_strategy([("DYNAMIC", 10)])
