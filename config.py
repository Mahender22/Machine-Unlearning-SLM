# config.py

# Base LLM model name (to be set in main.py)
BASE_LLM_MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'

# Device configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SLM model name
SLM_MODEL_NAME = 'distilbert-base-uncased'

# Paths for saving models and data
DATA_DIR = './data'
SLM_MODEL_DIR = './slm_model'

# Hugging Face authentication token (if required)
HF_AUTH_TOKEN = ''  # Set this in main.py if needed
