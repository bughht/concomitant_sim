from importlib.metadata import version
from .concomitant import concomitant_sim

__version__ = version("concomitant_sim")

__all__ = [
    "concomitant_sim",
]