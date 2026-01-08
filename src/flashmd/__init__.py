import warnings

from .models import get_pretrained as get_pretrained


warnings.filterwarnings("ignore", category=UserWarning, message="custom data")
