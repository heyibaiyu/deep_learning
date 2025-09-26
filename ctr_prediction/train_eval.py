
from  torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import MlpWithEmbedding, WideAndDeepV1, WideAndDeepV2, WideAndDeepV3, WideAndDeepV4, DeepFM
from utils import CriteoDataset, parse_criteo_csv, CriteoDatasetV2, CriteoDatasetOHE
import torch.nn as nn

import pandas as pd


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    for x_num, x_cat, y in val_loader:
        with torch.no_grad():
            predict = model(x_num, x_cat)
            # loss = torch.nn.functional.binary_cross_entropy(predict, y.float())
            loss = criterion(predict, y.float())
            val_loss += loss.item()
            all_preds.extend(predict.detach().numpy())
            all_labels.extend(y.numpy())
    model.train()
    val_auc = safe_auc(all_labels, all_preds)
    return val_loss / len(val_loader), val_auc


def evaluate_ohe(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    for x_num, x_cat, x_ohe, y in val_loader:
        with torch.no_grad():
            predict = model(x_num, x_cat,x_ohe)
            # loss = torch.nn.functional.binary_cross_entropy(predict, y.float())
            loss = criterion(predict, y.float())
            val_loss += loss.item()
            all_preds.extend(predict.detach().numpy())
            all_labels.extend(y.numpy())
    model.train()
    val_auc = safe_auc(all_labels, all_preds)
    return val_loss / len(val_loader), val_auc

def safe_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:  # happens if only one class present
        return float("nan")


def train(model, train_loader, val_loader, optimizer, num_epochs, ohe=False):
    model.train()
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')
    counter = 0
    patience = 2
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # dynamic lr
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,   # decrease lr by 50%
        patience=2,   # trigger lr adjustment if no loss decrease after x continuous epochs
        # verbose=True
    )

    print('-------- start training --------')
    print(' -------- DeepFM --------')
    for epoch in range(num_epochs):
        total_loss = 0
        all_preds, all_labels = [], []
        if not ohe:
            # no one hot encoding features
            for x_num, x_cat, y in train_loader:
                optimizer.zero_grad()
                predict = model(x_num, x_cat)
                # loss = torch.nn.functional.binary_cross_entropy(predict, y.float())
                loss = criterion(predict, y.float())
                total_loss += loss.item()
                all_preds.extend(predict.detach().numpy())
                all_labels.extend(y.numpy())
                loss.backward()
                optimizer.step()
        else:
            for x_num, x_cat, x_ohe, y in train_loader:
                optimizer.zero_grad()
                predict = model(x_num, x_cat, x_ohe)
                loss = criterion(predict, y.float())
                total_loss += loss.item()
                all_preds.extend(predict.detach().numpy())
                all_labels.extend(y.numpy())
                loss.backward()
                optimizer.step()


        epoch_loss = total_loss / len(train_loader)
        train_loss.append(epoch_loss)
        train_auc = safe_auc(all_labels, all_preds)
        if ohe:
            eval_loss, val_auc = evaluate_ohe(model, val_loader, criterion)
        else:
            eval_loss, val_auc = evaluate(model, val_loader, criterion)
        val_loss.append(eval_loss)

        lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch}, train_loss: {epoch_loss:.4f}, val_loss: {eval_loss:.4f}, train_auc: {train_auc:.4f}, val_auc: {val_auc:.4f}, lr: {lr:.6f}')

        # adjust lr
        scheduler.step(eval_loss)

        # early stop:
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
        else:
            counter += 1

        if counter >= patience:
            print('Early stopping')
            break


# Create dataloaders
def collate_fn(batch):
    x_num = torch.stack([b[0] for b in batch])
    x_cat = torch.stack([b[1] for b in batch])
    y = torch.stack([b[2] for b in batch])
    return x_num, x_cat, y

def collate_fn_ohe(batch):
    x_num = torch.stack([b[0] for b in batch])
    x_cat = torch.stack([b[1] for b in batch])
    x_ohe = torch.stack([b[2] for b in batch])
    y = torch.stack([b[3] for b in batch])
    return x_num, x_cat, x_ohe, y


def train_demo():
    # fake small dataset for example
    num_samples = 1000
    num_numeric = 13
    num_categories = [1000] * 26

    model = MlpWithEmbedding(num_numeric, num_categories)
    torch.manual_seed(123)
    x_num = torch.randn(num_samples, num_numeric)
    x_cat = torch.randint(0, 1000, (num_samples, 26))
    y = torch.randint(0, 2, (num_samples,))

    dataset = CriteoDataset(x_num, x_cat, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    train(model, dataloader, dataloader, optimizer, num_epochs=5)

if __name__ == '__main__':
    # train_demo()
    df, num_col, category_col = (parse_criteo_csv(
        './dac/train.txt', True, 13, 26, 1000000))  #1000000

    len_df = len(df)
    train_len = int(len_df*0.8)
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = df[:train_len]
    val_df = df[train_len:]

    # ohe = True # one hot encoding: only used for WideAndDeepV4
    ohe = False

    if ohe:
        ohe_column = pd.get_dummies(train_df[category_col], prefix=category_col)
        train_df_ohe = pd.concat([train_df, ohe_column], axis=1)
        train_ohe_columns = train_df_ohe.columns

        ohe_column = pd.get_dummies(val_df[category_col], prefix=category_col)
        val_df_ohe = pd.concat([val_df, ohe_column], axis=1)
        # reindex val to match train ohe columns
        val_df_ohe = val_df_ohe.reindex(columns=train_df_ohe.columns, fill_value=0)

        # ohe_columns = train_ohe_columns - num_col - category_col
        ohe_columns = [c for c in train_ohe_columns if c not in num_col and c not in category_col and c != 'label']
        print('train_df_ohe shape ', train_df_ohe.shape)


    if ohe:
        train_dataset = CriteoDatasetOHE(train_df_ohe, num_col, category_col, ohe_columns)
        test_dataset = CriteoDatasetOHE(val_df_ohe, num_col, category_col, ohe_columns, train_dataset.vocabs, train_dataset.scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn_ohe)
        test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn_ohe)
        # ohe_column_num = sum(train_dataset.vocab_size.values()) - 2 * len(train_dataset.vocab_size)
        # print('ohe_column_num:', ohe_column_num)
        model = WideAndDeepV4(len(num_col), [train_dataset.vocab_size[c] for c in category_col], len(ohe_columns))
        optimizer = torch.optim.Adam(model.parameters())
        train(model, train_dataloader, test_dataloader, optimizer, num_epochs=20, ohe=ohe)
    else:
        train_dataset = CriteoDatasetV2(train_df, num_col, category_col)
        test_dataset = CriteoDatasetV2(val_df, num_col, category_col, train_dataset.vocabs, train_dataset.scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)
        # model = MlpWithEmbedding(len(num_col), [train_dataset.vocab_size[c] for c in category_col])
        model = DeepFM(len(num_col), [train_dataset.vocab_size[c] for c in category_col])
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
        train(model, train_dataloader, test_dataloader, optimizer, num_epochs=20, ohe=ohe)





"""
data shape (2000000, 40)
Epoch 0, train_loss: 0.6618, val_loss: 0.4891, train_auc: 0.6599, val_auc: 0.7387
Epoch 1, train_loss: 0.4925, val_loss: 0.4763, train_auc: 0.7428, val_auc: 0.7570
Epoch 2, train_loss: 0.4808, val_loss: 0.4737, train_auc: 0.7636, val_auc: 0.7626
Epoch 3, train_loss: 0.4694, val_loss: 0.4755, train_auc: 0.7756, val_auc: 0.7588
Epoch 4, train_loss: 0.4535, val_loss: 0.4736, train_auc: 0.7935, val_auc: 0.7615
Epoch 5, train_loss: 0.4423, val_loss: 0.4796, train_auc: 0.8047, val_auc: 0.7561
Epoch 6, train_loss: 0.4280, val_loss: 0.4868, train_auc: 0.8198, val_auc: 0.7534
Epoch 7, train_loss: 0.4123, val_loss: 0.4952, train_auc: 0.8345, val_auc: 0.7496
Epoch 8, train_loss: 0.4036, val_loss: 0.5085, train_auc: 0.8434, val_auc: 0.7420
Epoch 9, train_loss: 0.3865, val_loss: 0.5180, train_auc: 0.8563, val_auc: 0.7428

data shape (1000000, 40)
Epoch 0, train_loss: 0.9024, val_loss: 0.5147, train_auc: 0.6274, val_auc: 0.7105
Epoch 1, train_loss: 0.5175, val_loss: 0.4929, train_auc: 0.7132, val_auc: 0.7437
Epoch 2, train_loss: 0.4918, val_loss: 0.4865, train_auc: 0.7494, val_auc: 0.7562
Epoch 3, train_loss: 0.4755, val_loss: 0.4871, train_auc: 0.7712, val_auc: 0.7539
Epoch 4, train_loss: 0.4687, val_loss: 0.4882, train_auc: 0.7818, val_auc: 0.7522
Epoch 5, train_loss: 0.4738, val_loss: 0.5005, train_auc: 0.7840, val_auc: 0.7424
Epoch 6, train_loss: 0.4426, val_loss: 0.4933, train_auc: 0.8063, val_auc: 0.7486
Epoch 7, train_loss: 0.4348, val_loss: 0.4938, train_auc: 0.8133, val_auc: 0.7445
Epoch 8, train_loss: 0.4173, val_loss: 0.5021, train_auc: 0.8330, val_auc: 0.7436
Epoch 9, train_loss: 0.3951, val_loss: 0.5216, train_auc: 0.8524, val_auc: 0.7386

# add numerical feature StandardScaler

data shape (1000000, 40)
-------- start training --------
Epoch 0, train_loss: 0.5034, val_loss: 0.4833, train_auc: 0.7289, val_auc: 0.7534
Epoch 1, train_loss: 0.4801, val_loss: 0.4764, train_auc: 0.7636, val_auc: 0.7632
Epoch 2, train_loss: 0.4649, val_loss: 0.4753, train_auc: 0.7833, val_auc: 0.7660
Epoch 3, train_loss: 0.4499, val_loss: 0.4769, train_auc: 0.8012, val_auc: 0.7647
Epoch 4, train_loss: 0.4341, val_loss: 0.4880, train_auc: 0.8184, val_auc: 0.7585
Epoch 5, train_loss: 0.4166, val_loss: 0.4958, train_auc: 0.8356, val_auc: 0.7532
Epoch 6, train_loss: 0.3993, val_loss: 0.5053, train_auc: 0.8509, val_auc: 0.7490
Epoch 7, train_loss: 0.3821, val_loss: 0.5234, train_auc: 0.8650, val_auc: 0.7419
Epoch 8, train_loss: 0.3652, val_loss: 0.5488, train_auc: 0.8777, val_auc: 0.7352
Epoch 9, train_loss: 0.3500, val_loss: 0.5629, train_auc: 0.8880, val_auc: 0.7302


data shape (1000000, 40)
-------- start training --------
 -------- early stop, weight_decay --------
Epoch 0, train_loss: 0.5029, val_loss: 0.4843, train_auc: 0.7297, val_auc: 0.7529
Epoch 1, train_loss: 0.4804, val_loss: 0.4776, train_auc: 0.7633, val_auc: 0.7617
Epoch 2, train_loss: 0.4656, val_loss: 0.4751, train_auc: 0.7825, val_auc: 0.7654
Epoch 3, train_loss: 0.4507, val_loss: 0.4756, train_auc: 0.8005, val_auc: 0.7667
Epoch 4, train_loss: 0.4347, val_loss: 0.4863, train_auc: 0.8180, val_auc: 0.7615
Epoch 5, train_loss: 0.4173, val_loss: 0.4922, train_auc: 0.8350, val_auc: 0.7568
Early stopping

# just early stop parameters
data shape (1000000, 40)
-------- start training --------
 -------- early stop, weight_decay, ReduceLROnPlateau --------
Epoch 0, train_loss: 0.5061, val_loss: 0.4846, train_auc: 0.7248, val_auc: 0.7525, lr: 0.001000
Epoch 1, train_loss: 0.4812, val_loss: 0.4786, train_auc: 0.7623, val_auc: 0.7630, lr: 0.001000
Epoch 2, train_loss: 0.4666, val_loss: 0.4756, train_auc: 0.7816, val_auc: 0.7655, lr: 0.001000
Epoch 3, train_loss: 0.4516, val_loss: 0.4771, train_auc: 0.7994, val_auc: 0.7652, lr: 0.001000
Epoch 4, train_loss: 0.4356, val_loss: 0.4833, train_auc: 0.8169, val_auc: 0.7616, lr: 0.001000
Early stopping



data shape (2000000, 40)
-------- start training --------
 -------- early stop, weight_decay, ReduceLROnPlateau, WideAndDeep --------
Epoch 0, train_loss: 187.5194, val_loss: 10.9658, train_auc: 0.5147, val_auc: 0.5468, lr: 0.001000
Epoch 1, train_loss: 12.2123, val_loss: 5.9619, train_auc: 0.5477, val_auc: 0.6153, lr: 0.001000
Epoch 2, train_loss: 10.8169, val_loss: 3.8596, train_auc: 0.5626, val_auc: 0.6152, lr: 0.001000
Epoch 3, train_loss: 11.6008, val_loss: 9.7381, train_auc: 0.5732, val_auc: 0.5831, lr: 0.001000
Epoch 4, train_loss: 9.0737, val_loss: 4.7941, train_auc: 0.5943, val_auc: 0.6178, lr: 0.001000
Epoch 5, train_loss: 7.4048, val_loss: 2.6746, train_auc: 0.6163, val_auc: 0.6420, lr: 0.001000
Epoch 6, train_loss: 7.1402, val_loss: 3.0975, train_auc: 0.6345, val_auc: 0.6699, lr: 0.001000
Early stopping


data shape (1000000, 40)
-------- start training --------
 -------- early stop, weight_decay, ReduceLROnPlateau, WideAndDeep --------
Epoch 0, train_loss: 145.3186, val_loss: 3.4569, train_auc: 0.5380, val_auc: 0.6238, lr: 0.001000
Epoch 1, train_loss: 6.6126, val_loss: 2.2474, train_auc: 0.5736, val_auc: 0.6386, lr: 0.001000
Epoch 2, train_loss: 6.7364, val_loss: 1.0829, train_auc: 0.5817, val_auc: 0.6765, lr: 0.001000
Epoch 3, train_loss: 5.8264, val_loss: 5.4479, train_auc: 0.5962, val_auc: 0.6791, lr: 0.001000
Epoch 4, train_loss: 4.1331, val_loss: 1.9271, train_auc: 0.6244, val_auc: 0.6379, lr: 0.001000
Epoch 5, train_loss: 3.8958, val_loss: 3.2335, train_auc: 0.6375, val_auc: 0.5974, lr: 0.001000
Epoch 6, train_loss: 3.3419, val_loss: 1.4551, train_auc: 0.6581, val_auc: 0.6887, lr: 0.001000
Epoch 7, train_loss: 2.9463, val_loss: 2.0350, train_auc: 0.6790, val_auc: 0.6803, lr: 0.001000
Early stopping


data shape (1000000, 40)
-------- start training --------
 -------- early stop, weight_decay, ReduceLROnPlateau, WideAndDeep, patience from 3 to 8 --------
Epoch 0, train_loss: 246.2501, val_loss: 23.1281, train_auc: 0.5497, val_auc: 0.6129, lr: 0.001000
Epoch 1, train_loss: 10.2215, val_loss: 4.3550, train_auc: 0.5564, val_auc: 0.5921, lr: 0.001000
Epoch 2, train_loss: 8.4166, val_loss: 3.4582, train_auc: 0.5654, val_auc: 0.6572, lr: 0.001000
Epoch 3, train_loss: 6.0966, val_loss: 1.8095, train_auc: 0.5794, val_auc: 0.6283, lr: 0.001000
Epoch 4, train_loss: 5.9943, val_loss: 1.8040, train_auc: 0.5900, val_auc: 0.6304, lr: 0.001000
Epoch 5, train_loss: 5.1081, val_loss: 3.0591, train_auc: 0.6059, val_auc: 0.6307, lr: 0.001000
Epoch 6, train_loss: 4.8441, val_loss: 3.0270, train_auc: 0.6212, val_auc: 0.6846, lr: 0.001000
Epoch 7, train_loss: 4.5069, val_loss: 1.4676, train_auc: 0.6388, val_auc: 0.6570, lr: 0.001000
Epoch 8, train_loss: 2.9960, val_loss: 1.3381, train_auc: 0.6624, val_auc: 0.6777, lr: 0.001000
Epoch 9, train_loss: 3.2305, val_loss: 2.3445, train_auc: 0.6717, val_auc: 0.6368, lr: 0.001000
Epoch 10, train_loss: 2.8370, val_loss: 2.0042, train_auc: 0.6896, val_auc: 0.6777, lr: 0.001000
Epoch 11, train_loss: 2.5556, val_loss: 3.3072, train_auc: 0.7045, val_auc: 0.6766, lr: 0.001000
Epoch 12, train_loss: 2.3293, val_loss: 2.8045, train_auc: 0.7218, val_auc: 0.6597, lr: 0.001000
Epoch 13, train_loss: 2.7142, val_loss: 3.8112, train_auc: 0.7265, val_auc: 0.6507, lr: 0.001000
Epoch 14, train_loss: 2.3588, val_loss: 4.7761, train_auc: 0.7416, val_auc: 0.6510, lr: 0.001000
Early stopping


 -------- early stop, weight_decay, ReduceLROnPlateau, WideAndDeepV4, one hot encoding (10k training data) --------
 -------- early stop, weight_decay, ReduceLROnPlateau, WideAndDeepV4 --------
Epoch 0, train_loss: 0.6082, val_loss: 0.5247, train_auc: 0.5257, val_auc: 0.5974, lr: 0.001000
Epoch 1, train_loss: 0.5152, val_loss: 0.5099, train_auc: 0.6207, val_auc: 0.6779, lr: 0.001000
Epoch 2, train_loss: 0.4826, val_loss: 0.5038, train_auc: 0.7333, val_auc: 0.7139, lr: 0.001000
Epoch 3, train_loss: 0.4614, val_loss: 0.4923, train_auc: 0.7746, val_auc: 0.7140, lr: 0.001000
Epoch 4, train_loss: 0.4472, val_loss: 0.4867, train_auc: 0.7848, val_auc: 0.7178, lr: 0.001000
Epoch 5, train_loss: 0.4306, val_loss: 0.4841, train_auc: 0.8151, val_auc: 0.7198, lr: 0.001000
Epoch 6, train_loss: 0.4181, val_loss: 0.4839, train_auc: 0.8251, val_auc: 0.7215, lr: 0.001000
Epoch 7, train_loss: 0.4031, val_loss: 0.4830, train_auc: 0.8416, val_auc: 0.7226, lr: 0.001000
Epoch 8, train_loss: 0.3875, val_loss: 0.4846, train_auc: 0.8603, val_auc: 0.7229, lr: 0.001000
Epoch 9, train_loss: 0.3717, val_loss: 0.4880, train_auc: 0.8752, val_auc: 0.7203, lr: 0.001000
Early stopping



data shape (50000, 40)
train_df_ohe shape  (40000, 119609)
-------- start training --------
 -------- early stop, weight_decay, ReduceLROnPlateau, WideAndDeepV4 (50k training data) --------
Epoch 0, train_loss: 0.5225, val_loss: 0.4878, train_auc: 0.6143, val_auc: 0.7197, lr: 0.001000
Epoch 1, train_loss: 0.4660, val_loss: 0.4738, train_auc: 0.7529, val_auc: 0.7310, lr: 0.001000
Epoch 2, train_loss: 0.4408, val_loss: 0.4700, train_auc: 0.7896, val_auc: 0.7375, lr: 0.001000
Epoch 3, train_loss: 0.4211, val_loss: 0.4688, train_auc: 0.8170, val_auc: 0.7422, lr: 0.001000
Epoch 4, train_loss: 0.3980, val_loss: 0.4733, train_auc: 0.8392, val_auc: 0.7377, lr: 0.001000


data shape (1000000, 40)
-------- start training --------
 -------- DeepFM --------
Epoch 0, train_loss: 31.5386, val_loss: 12.8867, train_auc: 0.5970, val_auc: 0.6374, lr: 0.001000
Epoch 1, train_loss: 12.8392, val_loss: 10.4614, train_auc: 0.6366, val_auc: 0.6420, lr: 0.001000
Epoch 2, train_loss: 10.5593, val_loss: 9.7349, train_auc: 0.6525, val_auc: 0.6402, lr: 0.001000
Epoch 3, train_loss: 9.2115, val_loss: 9.4656, train_auc: 0.6668, val_auc: 0.6455, lr: 0.001000
Epoch 4, train_loss: 8.0823, val_loss: 9.7521, train_auc: 0.6809, val_auc: 0.6475, lr: 0.001000
Epoch 5, train_loss: 7.1751, val_loss: 9.1231, train_auc: 0.6940, val_auc: 0.6411, lr: 0.001000
Epoch 6, train_loss: 6.6844, val_loss: 9.6860, train_auc: 0.7078, val_auc: 0.6428, lr: 0.001000
Early stopping



data shape (1000000, 40), emb from 12 to 32, add weight_decay for L2 regularization, change hidden layer size
-------- start training --------
 -------- DeepFM --------
Epoch 0, train_loss: 105.3346, val_loss: 31.5538, train_auc: 0.5909, val_auc: 0.6423, lr: 0.001000
Epoch 1, train_loss: 33.3792, val_loss: 23.7683, train_auc: 0.6360, val_auc: 0.6575, lr: 0.001000
Epoch 2, train_loss: 24.1684, val_loss: 20.3555, train_auc: 0.6500, val_auc: 0.6576, lr: 0.001000
Epoch 3, train_loss: 20.8479, val_loss: 17.7893, train_auc: 0.6559, val_auc: 0.6591, lr: 0.001000

○ V1: Epoch 14, train_loss: 2.3588, val_loss: 4.7761, train_auc: 0.7416, val_auc: 0.6510 (so low), lr: 0.001000
    § Val auc is much lower than simple MLP. The reason is wide part use raw categorical id as feature (big int)
○ V2: Remove categorical feature from wide component
    § Epoch 3, train_loss: 0.4504, val_loss: 0.4772, train_auc: 0.8007, val_auc: 0.7659(recovered, but didn't improve too much), lr: 0.001000
○ V3: Use same feature as deep component in wide layer
    § Epoch 2, train_loss: 0.4644, val_loss: 0.4762, train_auc: 0.7836, val_auc: 0.7650 (wide doesn't learn anything from emb), lr: 0.001000
○ V4: V2 + bias=True in wide layer
    Epoch 3, train_loss: 0.4523, val_loss: 0.4741, train_auc: 0.7986, val_auc: 0.7670 (better), lr: 0.001000
○ V5-1 (10k data): one-hot-encoding categorical features and use them in wide layer together with numerical features. 
    Epoch 8, train_loss: 0.3875, val_loss: 0.4846, train_auc: 0.8603, val_auc: 0.7229 (better than V1, but worse than V2), lr: 0.001000
  V5-2 (50k data): 
    Epoch 3, train_loss: 0.4211, val_loss: 0.4688, train_auc: 0.8170, val_auc: 0.7422, lr: 0.001000

"""

