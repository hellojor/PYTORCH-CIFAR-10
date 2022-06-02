import torch
import torchvision.transforms as transforms

import os

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameter
configure = {
    'lr' : 1e-3,
    'num_epoch' : 200,
    'batch_size' : 8,
    'early_stop' : 20,
    'transform' : transforms.Compose( 
            [transforms.ToTensor(), # [0,1]
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
            ]
            )
    
}