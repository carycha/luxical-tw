import logging
from .__about__ import __version__
from .embedder import Model, Embedder
from .trainer import Trainer

logger = logging.getLogger(__name__)

__all__ = ["__version__", "Model", "Embedder", "Trainer", "init"]

def init(user_dict_path: str | None = None):
    """
    Initialize Luxical global configuration.
    
    Args:
        user_dict_path: Optional path to a standard jieba-format text dictionary.
    """
    if user_dict_path:
        import jieba_fast_dat
        logger.info(f"Initializing Luxical with user dictionary: {user_dict_path}")
        jieba_fast_dat.load_userdict(str(user_dict_path))
    else:
        logger.info("Initializing Luxical with default settings.")
