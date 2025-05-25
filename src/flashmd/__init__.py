import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="custom data")

from .models import get_universal_model
