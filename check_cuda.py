import torch
import os

# Check Cuda
print("\nIs CUDA supported by this system?","\tYes" if torch.cuda.is_available() else "\tNo")
print('PyTorch version:\t\t\t', torch.__version__)
print('CUDA version:\t\t\t\t', torch.version.cuda)
print('cuDNN version:\t\t\t\t', torch.backends.cudnn.version())

# Check CPU Count
print(f"Number of CPU's: \t\t\t{os.cpu_count()}")


# Storing ID of current CUDA device
if torch.cuda.is_available():
    cuda_id = torch.cuda.current_device()
    cuda_device = torch.cuda.get_device_name()
    print(f"ID of current CUDA device:\t\t{cuda_id}")
    print(f"Name of current CUDA device:\t\t{cuda_device}")
    torch.cuda.empty_cache()

