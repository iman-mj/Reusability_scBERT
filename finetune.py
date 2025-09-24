# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=20, help='Number of epochs.') # Increased epochs for meaningful training
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default= 3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='/home/iman/scBERT/data/Zheng68K.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='/home/iman/scBERT/data/panglao_pretrain.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='/home/iman/scBERT/data/Saved_model/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetuNeneurips', help='Finetuned model name.')
args = parser.parse_args()

log_file = "/home/iman/scBERT/data/neurips_validation_report.txt"

# --- Distributed Setup (Correct Order) ---
local_rank = int(os.environ["LOCAL_RANK"])
is_master = local_rank == 0
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

# --- Hyperparameters ---
SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
PATIENCE = 10
UNASSIGN_THRES = 0.0
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
model_name = args.model_name
ckpt_dir = args.ckpt_dir

seed_all(SEED + torch.distributed.get_rank())

# --- Dataset Class (FIXED) ---
'''
class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]
'''

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        # FIX 1: Use the 'index' provided by the DataLoader, not a random one.
        #rand_start = random.randint(0, self.data.shape[0]-1)
        #full_seq = self.data[rand_start].toarray()[0]
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()

        # FIX 2: Do NOT move to 'device' here. This will be done in the training loop.
        full_seq = torch.cat((full_seq, torch.tensor([0])))

        seq_label = self.label[index]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

"""
# --- Simple Classifier Head ---
class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        # The PerformerLM output is 2D (batch, seq_len), so we don't need convolutions
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
"""
"""
class Identity(torch.nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super(Identity, self).__init__()
        # FIX 1: The first layer must accept the 200 features
        # from the Performer model's output.
        self.fc1 = nn.Linear(in_features=200, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        # The PerformerLM outputs a tensor of shape (batch, sequence_length, features).
        # We take the features from only the FIRST token for classification.
        # This is a standard practice for models like BERT.
        x = x[:, 0, :]

        # Now the tensor 'x' has the correct shape (batch, 200)
        # to be processed by our linear layers.
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
"""


# --- Data Loading ---
if is_master:
    print("Loading data...")
data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(np.array(data.obs['cell_type']), return_inverse=True)
with open('/home/iman/scBERT/code/neurips_label.pkl', 'wb') as fp:
    pkl.dump(label, fp)
label = torch.from_numpy(label)
data = data.X

if is_master:
    # Save label dictionary only from the master process
    with open('/home/iman/scBERT/code/neurips_label_dict.pkl', 'wb') as fp:
        pkl.dump(label_dict, fp)

# --- Train/Validation Split and Main Loop ---
print("learning started")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = POS_EMBED_USING
)

path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = False
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)

dist.barrier()
trigger_times = 0
max_acc = 0.0

for i in range(1, EPOCHS+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {i}/{EPOCHS + 1}"):
        index += 1
        data, labels = data.to(device), labels.to(device)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                logits = model(data)
                loss = loss_fn(logits, labels)
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    dist.barrier()
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                #final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                final[np.amax(final_prob.cpu().numpy(), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)
            del data_v, labels_v, logits, final_prob, final
            # gather
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            no_drop = predictions != -1
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())
            cur_acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average='macro')
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
            if is_master:
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==')
                print(confusion_matrix(truths, predictions))
                print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))
                with open(log_file, "a") as f:
                        print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==', file=f)
                        print(confusion_matrix(truths, predictions), file=f)
                        print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4), file=f)
            if cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0
                save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
            else:
                trigger_times += 1
                if trigger_times > PATIENCE:
                    break
    del predictions, truths


'''
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
for index_train, index_val in sss.split(data, label):

    # --- Setup Datasets and DataLoaders ---
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)
    print("I'm hear (1)")

    if is_master:
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Model Setup ---
    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = POS_EMBED_USING
    )
    print("I'm hear (2)")

    # Load pretrained weights
    path = args.model_path
    ckpt = torch.load(path, map_location='cpu') # Load to CPU first
    model.load_state_dict(ckpt['model_state_dict'])

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = False
    for param in model.performer.net.layers[-2].parameters():
        param.requires_grad = False

    # Attach new classifier head
    model.to_out = Identity(dropout=0.5, h_dim=128, out_dim=label_dict.shape[0])
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- Optimizer, Scheduler, Loss ---
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=15, cycle_mult=2, max_lr=LEARNING_RATE,
        min_lr=1e-6, warmup_steps=10, gamma=0.9
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    # --- Training Loop ---
    dist.barrier()
    trigger_times = 0
    max_f1 = 0.0
    print("I'm hear (3)")
    for i in range(1, EPOCHS + 1):
        train_loader.sampler.set_epoch(i)
        model.train()
        running_loss = 0.0
        cum_acc = 0.0
        print("I'm hear (3-1)")
        for index, (batch_data, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {i}/{EPOCHS + 1}"):
            batch_data, labels = batch_data.to(device), labels.to(device)
            
            # Forward pass
            logits = model(batch_data)
            loss = loss_fn(logits, labels)
            
            # Normalize loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION
            #print(f"losss is {loss}")
            loss.backward()

            if (index + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * GRADIENT_ACCUMULATION # Un-normalize for logging
            
            # Accuracy calculation
            final = logits.argmax(dim=-1)
            correct_num = torch.eq(final, labels).sum().item()
            cum_acc += correct_num / labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * cum_acc / len(train_loader)
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
        print("I'm hear (4)")
        if is_master:
            print(f'== Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}% ==')
        
        dist.barrier()
        scheduler.step()

        # --- Validation Loop ---
        if i % VALIDATE_EVERY == 0:
            model.eval()
            predictions = []
            truths = []
            with torch.no_grad():
                for index, (data_v, labels_v) in enumerate(val_loader):
                    data_v, labels_v = data_v.to(device), labels_v.to(device)
                    logits = model(data_v)
                    final = logits.argmax(dim=-1)
                    predictions.append(final)
                    truths.append(labels_v)

            # Gather results from all GPUs
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            
            if is_master:
                predictions = predictions.cpu().numpy()
                truths = truths.cpu().numpy()
                cur_f1 = f1_score(truths, predictions, average='macro')
                print(f'== Epoch: {i} | Validation F1 Score: {cur_f1:.6f} ==')
                # print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))

                if cur_f1 > max_f1:
                    max_f1 = cur_f1
                    trigger_times = 0
                    save_best_ckpt(i, model, optimizer, scheduler, cur_f1, model_name, ckpt_dir)
                else:
                    trigger_times += 1
                    if trigger_times > PATIENCE:
                        if is_master:
                            print("Early stopping!")
                        break
        if trigger_times > PATIENCE:
            break

'''