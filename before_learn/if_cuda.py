import torch
   
   
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")