import os, time, datetime, warnings

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from model.nicheTrans_img import *
from datasets.data_manager_SMA import SMA

from utils.utils import *
from utils.utils_training_SMA import train, test
from utils.utils_dataloader import *

warnings.filterwarnings("ignore")

# Initialize args and fix seeds
exec(open('./args/args_SMA.py').read())

set_seed(args.seed)
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))
print("==========\nArgs:{}\n==========".format(args))

# Initialize dataloaders and NicheTrans
dataset = SMA(path_img=args.path_img, rna_path=args.rna_path, msi_path=args.msi_path,
              n_top_genes=args.n_source, n_top_targets=args.n_target,
              cell_type_visualize=args.cell_type_visualize,
              cell_type_visualization_dir=args.cell_type_visualization_dir,
              cell_type_visualization_dpi=args.cell_type_visualization_dpi)
trainloader, testloader = sma_dataloader(args, dataset)

source_dimension, target_dimension = dataset.rna_length, dataset.msi_length
model = NicheTrans(source_length=source_dimension, target_length=target_dimension,
                   noise_rate=args.noise_rate, dropout_rate=args.dropout_rate,
                   n_spot_types=dataset.n_spot_types)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Initialize loss function and optimizer
criterion = nn.MSELoss()

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
else:
    print('unexpected optimizer')

if args.stepsize > 0:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

# Training and testing
start_time = time.time()

for epoch in range(args.max_epoch):
    print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
    train(model, criterion, optimizer, trainloader, use_img=False, device=device)
    if args.stepsize > 0:
        scheduler.step()

pearson = test(model, testloader, use_img=False, device=device)
torch.save(model.state_dict(), 'NicheTrans_SMA_last.pth')

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
