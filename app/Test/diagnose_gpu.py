# diagnostic_gpu.py
import torch

print("CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Memory allocated:", torch.cuda.memory_allocated(0) / 1024**3, "GB")
    print("Max memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("No CUDA – running on CPU. That's why it's slow.")
    
    
