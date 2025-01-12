"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *


ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))

ASSETS_DATA_DIR = os.path.abspath(os.path.join(ASSETS_EXT_DIR, "data"))

