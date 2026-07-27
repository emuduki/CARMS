from pathlib import Path
from typing import Optional

import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.special import logsumexp

from src.utils.logger import get_logger
from src.utils.data_splits import get_test_start
from src.regime.constants import REGIME_NAMES, REGIME_COLOURS
from src.regime.regime_analyser import analyse_regimes

log = get_logger(__name__)

# (all remaining code remains the same as previous implementation)
