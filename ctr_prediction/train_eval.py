
from  torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary

from model import MlpWithEmbedding, WideAndDeepV1, WideAndDeepV2, WideAndDeepV3, WideAndDeepV4, DeepFM, DCN, DcnV2
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


def train(model, train_loader, val_loader, optimizer, num_epochs, ohe=False, patience=5):
    model.train()
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')
    counter = 0
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # dynamic lr
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,   # decrease lr by 50%
        patience=patience,   # trigger lr adjustment if no loss decrease after x continuous epochs
        # verbose=True
    )

    print('-------- start training --------')
    print(' -------- DCN V2 --------')
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


def get_input_size(dataloader):
    x_num, x_cat, y = next(iter(dataloader))
    return [x_num.shape, x_cat.shape]


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


    lr = 0.001
    weight_decay = 5e-4
    rank = 32
    patience = 4
    if ohe:
        train_dataset = CriteoDatasetOHE(train_df_ohe, num_col, category_col, ohe_columns)
        test_dataset = CriteoDatasetOHE(val_df_ohe, num_col, category_col, ohe_columns, train_dataset.vocabs, train_dataset.scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn_ohe)
        test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn_ohe)
        # ohe_column_num = sum(train_dataset.vocab_size.values()) - 2 * len(train_dataset.vocab_size)
        # print('ohe_column_num:', ohe_column_num)
        model = WideAndDeepV4(len(num_col), [train_dataset.vocab_size[c] for c in category_col], len(ohe_columns))
        optimizer = torch.optim.Adam(model.parameters())
        train(model, train_dataloader, test_dataloader, optimizer, num_epochs=20, ohe=ohe, patience=patience)
    else:
        train_dataset = CriteoDatasetV2(train_df, num_col, category_col)
        test_dataset = CriteoDatasetV2(val_df, num_col, category_col, train_dataset.vocabs, train_dataset.scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)
        # model = MlpWithEmbedding(len(num_col), [train_dataset.vocab_size[c] for c in category_col])
        # model = DeepFM(len(num_col), [train_dataset.vocab_size[c] for c in category_col])
        model = DcnV2(len(num_col), [train_dataset.vocab_size[c] for c in category_col], rank=rank)
        summary(model, input_size=get_input_size(train_dataloader))
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
        train(model, train_dataloader, test_dataloader, optimizer, num_epochs=20, ohe=ohe, patience=patience)

    print('Hyper parameters: \r', f"lr: {lr: .6f}", f"weight_decay: {weight_decay: .6f}", f"rank: {rank: d}", f"patience: {patience}")

