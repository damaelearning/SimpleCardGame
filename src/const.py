from pathlib import Path
import os

current_dir = Path(__file__).parent

#The path to directory storing tensorflow model file
MODEL_DIR = current_dir/'../model'

#The path to directory storing game history file
HISTORY_DIR = current_dir/'../history'
