"""Global variables."""
import os

#
# Folder paths

APP_DIR = os.getcwd()
DATA_DIR = os.path.join(APP_DIR, 'data/')

#
# Numerical stability

EPS = 1e-6
