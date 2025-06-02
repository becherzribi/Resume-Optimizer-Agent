# check_cuda.py

import torch

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU.")

if __name__ == "__main__":
    main()
