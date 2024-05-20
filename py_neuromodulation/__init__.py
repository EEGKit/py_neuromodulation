from .nm_logger import NMLogger
from pathlib import PurePath
from importlib.metadata import version

__version__ = version("py_neuromodulation")

# Define constant for py_nm directory
PYNM_DIR = PurePath(__file__).parent

# logger initialization first to prevent circular import
logger = NMLogger(__name__)

# Public API
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor
from .nm_settings import NMSettings
