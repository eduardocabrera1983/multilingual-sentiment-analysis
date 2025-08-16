import transformers
import torch
import streamlit as st
import pandas as pd
from transformers import pipeline
import langdetect

print("✅ All libraries imported successfully!")
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")