import pandas as pd
import torch
from  torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np

class CriteoDataset(Dataset):
    def __init__(self, x_num, x_cat, y):
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]

class CriteoDatasetV2(Dataset):
    def __init__(self, df, num_columns, cat_columns, vocabs=None, train_scaler=None):
        self.df = df.reset_index(drop=True)
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.vocabs = vocabs or {}
        self.scaler = train_scaler or StandardScaler()

        if not train_scaler:
            df_num_scaled = self.scaler.fit_transform(self.df[num_columns])
        else:
            df_num_scaled = self.scaler.transform(self.df[num_columns])

        # do not forget columns=num_columns otherwise the new df will not have column name
        df_num_scaled = pd.DataFrame(df_num_scaled, columns=num_columns, index=self.df.index)

        df_category = self.df[cat_columns]
        df_label = self.df[["label"]]
        self.df = pd.concat([df_label, df_num_scaled, df_category], axis=1)

        for c in cat_columns:
            if c not in self.vocabs:
                unique_words = self.df.loc[:, c].unique()
                self.vocabs[c] = {k: i + 2 for i, k in enumerate(unique_words)}
                # vocabs = {'cate_1': {'val1': 2, 'val2': 3, ..., 'valk': k+1}, 'cate_2': {}}
        self.vocab_size = {c: len(vocab) + 2 for c , vocab in self.vocabs.items()}
        # vocab_size = {'cate_1': 392, 'cate_2': 120, ..., 'cate_n': 201}


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_num = torch.tensor([row[c] for c in self.num_columns], dtype=torch.float32)
        # embedding input should be the vocab index of long type, 1 is default value when not found
        x_cat = torch.tensor([self.vocabs[c].get(str(row[c]), 1) for c in self.cat_columns], dtype=torch.long)
        y = torch.tensor([row['label']], dtype=torch.float32)
        return x_num, x_cat, y

class CriteoDatasetOHE(Dataset):
    # with one hot encoding
    def __init__(self, df, num_columns, cat_columns, ohe_columns, vocabs=None, train_scaler=None):
        self.df = df.reset_index(drop=True)
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.ohe_columns = ohe_columns
        self.vocabs = vocabs or {}
        self.scaler = train_scaler or StandardScaler()

        if not train_scaler:
            df_num_scaled = self.scaler.fit_transform(self.df[num_columns])
        else:
            df_num_scaled = self.scaler.transform(self.df[num_columns])

        # do not forget columns=num_columns otherwise the new df will not have column name
        df_num_scaled = pd.DataFrame(df_num_scaled, columns=num_columns, index=self.df.index)

        df_category = self.df[cat_columns]
        df_label = self.df[["label"]]
        df_ohe = self.df[ohe_columns]
        self.df = pd.concat([df_label, df_num_scaled, df_category, df_ohe], axis=1)

        for c in cat_columns:
            if c not in self.vocabs:
                unique_words = self.df.loc[:, c].unique()
                self.vocabs[c] = {k: i + 2 for i, k in enumerate(unique_words)}
                # vocabs = {'cate_1': {'val1': 2, 'val2': 3, ..., 'valk': k+1}, 'cate_2': {}}
        self.vocab_size = {c: len(vocab) + 2 for c , vocab in self.vocabs.items()}
        # vocab_size = {'cate_1': 392, 'cate_2': 120, ..., 'cate_n': 201}


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_num = torch.tensor([row[c] for c in self.num_columns], dtype=torch.float32)
        # embedding input should be the vocab index of long type, 1 is default value when not found
        x_cat = torch.tensor([self.vocabs[c].get(str(row[c]), 1) for c in self.cat_columns], dtype=torch.long)
        x_ohe = torch.tensor([row[c] for c in self.ohe_columns], dtype=torch.float32)

        y = torch.tensor([row['label']], dtype=torch.float32)
        return x_num, x_cat, x_ohe, y

def parse_criteo_csv(path, is_train, num_numeric, num_category, num_rows):
    num_col = ['num_' + str(i) for i in range(num_numeric)]
    category_col = ['cat_' + str(i) for i in range(num_category)]

    if is_train:
        # training file has label at first column, test not
        columns = ['label'] +  num_col + category_col
    else:
        columns = num_col + category_col

    df = pd.read_csv(path, sep="\t", header=None, names=columns, nrows=num_rows)
    print('data shape', df.shape)

    df['label'] = df['label'].astype(int)
    for c in num_col:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    for c in category_col:
        # df[c] = df[c].astype(str).replace({"nan": "__missing__", "": "__missing__"} )
        df[c] = df[c].fillna('__missing__')

    return df, num_col, category_col



if __name__ == '__main__':
    df, num_col, category_col = (parse_criteo_csv(
            './dac/train.txt', True, 13, 26, 3))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = CriteoDataset(train_df)
