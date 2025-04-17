import sys
print("Python Path:", sys.executable)

import torch
print("CUDA Available:", torch.cuda.is_available())
