import torch
# Check Cuda
print("\nIs CUDA supported by this system?","\tYes" if torch.cuda.is_available() else "\tNo")
print(f"CUDA version:\t\t\t\t{torch.version.cuda}")

# Storing ID of current CUDA device
if torch.cuda.is_available():
    cuda_id = torch.cuda.current_device()
    cuda_device = torch.cuda.get_device_name()
    print(f"ID of current CUDA device:\t\t{cuda_id}")
    print(f"Name of current CUDA device:\t\t{cuda_device}")
    torch.cuda.empty_cache()
