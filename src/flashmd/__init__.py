import warnings

from .models import get_pretrained as get_pretrained
from .models import save_checkpoint as save_checkpoint


warnings.filterwarnings("ignore", category=UserWarning, message="custom data")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`torch.jit.script`")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`torch.jit.load`")
