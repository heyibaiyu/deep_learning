from torch import nn
import torch

class MlpWithEmbedding(nn.Module):
    # 2 layer MLP, using embedding layer for categorical features
    def __init__(self, num_numeric, num_categories, emb_dim=12, hidden_dims=[128, 128, 64]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_category, emb_dim) for num_category in num_categories
        ])

        input_dim = num_numeric + len(num_categories) * emb_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)


    def forward(self, x_num, x_categories):
        embs = [emb(x_categories[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(embs, dim=-1)
        x = torch.cat([x_num, embs], dim=-1)
        out = torch.sigmoid(self.mlp(x))
        return out


class FactorizationMachine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1).pow(2)
        sum_of_square = torch.sum(x.pow(2), dim=1)
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)


class DeepFM(nn.Module):
    def __init__(self, num_numeric, num_categories, emb_dim=32, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.emb_dim = emb_dim
        # FM linear component
            # input: categorical ids, numerical features
            # output: scaler: w * x
        # vocab_size = sum(num_categories)
        # self.linear_embedding = nn.Embedding(vocab_size, 1)    # learn a scaler weight for each category id, w * x
        self.linear_embedding = nn.ModuleList([
            nn.Embedding(num_cat, 1) for num_cat in num_categories
        ])
        self.linear_num = nn.Linear(num_numeric, 1, bias=False)
        self.linear_bias = nn.Parameter(torch.zeros(1))

        # FM 2nd order interaction component
        self.fm = FactorizationMachine()
        self.num_embeddings = nn.Parameter(torch.randn(num_numeric, emb_dim))

        # Deep component
            # note: it shares the emb layer with FM 2nd order component
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_category, emb_dim) for num_category in num_categories
        ])

        input_dim = num_numeric + len(num_categories) * emb_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.deep_mlp = nn.Sequential(*layers)


    def forward(self, x_num, x_categories):
        linear_cats = [emb(x_categories[:, i]) for i, emb in enumerate(self.linear_embedding)]
        linear_cats = torch.sum(torch.cat(linear_cats, dim=1), dim=1, keepdim=True)
        p_linear = self.linear_bias + self.linear_num(x_num) + linear_cats

        cat_emb = [emb(x_categories[:, i]) for i, emb in enumerate(self.embeddings)]  #[(batch, emb_dim), (), ()]
        cat_emb = torch.stack(cat_emb, dim=1)  # convert 3 tensors in a list to one single tensor:  (batch, num_categorical, emb_dim)
        # num_emb = x_num.unsqueeze(2)     # convert tensor of shape (batch, num_numerical) to shape (batch, num_numerical, 1)
        # num_emb = num_emb.repeat(1, 1, self.emb_dim)   # make the last dimension shape same as cat_emb

        num_emb = x_num.unsqueeze(2) * self.num_embeddings.unsqueeze(0) # Note: this is import to higher AUC
        x_emb = torch.cat([cat_emb, num_emb], dim=1)
        p_fm_2d = self.fm(x_emb)

        dnn_input_embeddings = cat_emb.view(cat_emb.size(0), -1) # flatten embedding to (batch_size, -1)
        dnn_input_embeddings = torch.cat([dnn_input_embeddings, x_num], dim=1)
        p_dnn = self.deep_mlp(dnn_input_embeddings)
        out = p_linear + p_fm_2d + p_dnn
        return out



class WideAndDeepV1(nn.Module):
    # V1: use raw categorical big int id as feature, metric is worse
    def __init__(self, num_numeric, num_categories, emb_dim=12, hidden_dims=[128, 128, 64]):
        super().__init__()
        # deep component
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_category, emb_dim) for num_category in num_categories
        ])

        input_dim = num_numeric + len(num_categories) * emb_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.deep_mlp = nn.Sequential(*layers)

        # wide layer: both numeric and raw categorical features (integer-encoded categories) as input
        wide_dim = num_numeric + len(num_categories)
        self.wide_layer = nn.Linear(wide_dim, 1, bias=False)


    def forward(self, x_num, x_categories):
        embs = [emb(x_categories[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(embs, dim=-1)
        x_deep = torch.cat([x_num, embs], dim=-1)
        deep_out = self.deep_mlp(x_deep)
        x_wide = torch.cat([x_num, x_categories.float()], dim=-1)
        wide_out = self.wide_layer(x_wide)

        # sum the output from wide and deep
        out = wide_out + deep_out
        # out = torch.sigmoid(out)
        return out


class WideAndDeepV2(nn.Module):
    # V2: remove categorical features since it will be covered in deep component
    def __init__(self, num_numeric, num_categories, emb_dim=12, hidden_dims=[128, 128, 64]):
        super().__init__()
        # deep component
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_category, emb_dim) for num_category in num_categories
        ])

        input_dim = num_numeric + len(num_categories) * emb_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.deep_mlp = nn.Sequential(*layers)

        # wide layer: both numeric and raw categorical features (integer-encoded categories) as input
        wide_dim = num_numeric
        self.wide_layer = nn.Linear(wide_dim, 1, bias=True)


    def forward(self, x_num, x_categories):
        embs = [emb(x_categories[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(embs, dim=-1)
        x_deep = torch.cat([x_num, embs], dim=-1)
        deep_out = self.deep_mlp(x_deep)
        wide_out = self.wide_layer(x_num)

        # sum the output from wide and deep
        out = wide_out + deep_out
        # out = torch.sigmoid(out)
        return out


class WideAndDeepV3(nn.Module):
    # V3: add categorical embedding in wide layer, so wide and deep has same input
    def __init__(self, num_numeric, num_categories, emb_dim=12, hidden_dims=[128, 128, 64]):
        super().__init__()
        # deep component
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_category, emb_dim) for num_category in num_categories
        ])

        input_dim = num_numeric + len(num_categories) * emb_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.deep_mlp = nn.Sequential(*layers)

        # wide layer: both numeric and raw categorical features (integer-encoded categories) as input
        wide_dim = num_numeric + len(num_categories) * emb_dim
        self.wide_layer = nn.Linear(wide_dim, 1, bias=False)


    def forward(self, x_num, x_categories):
        embs = [emb(x_categories[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(embs, dim=-1)
        x_deep = torch.cat([x_num, embs], dim=-1)
        deep_out = self.deep_mlp(x_deep)
        wide_out = self.wide_layer(x_deep)

        # sum the output from wide and deep
        out = wide_out + deep_out
        # out = torch.sigmoid(out)
        return out


class WideAndDeepV4(nn.Module):
    # V4: use one hot encoding categorical feature in wide layer
    def __init__(self, num_numeric, num_categories, num_ohe, emb_dim=12, hidden_dims=[128, 128, 64]):
        super().__init__()
        # deep component
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_category, emb_dim) for num_category in num_categories
        ])

        input_dim = num_numeric + len(num_categories) * emb_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.deep_mlp = nn.Sequential(*layers)

        # wide layer: both numeric and raw categorical features (integer-encoded categories) as input
        wide_dim = num_numeric + num_ohe
        self.wide_layer = nn.Linear(wide_dim, 1, bias=True)


    def forward(self, x_num, x_categories, x_ohe):
        embs = [emb(x_categories[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(embs, dim=-1)
        x_deep = torch.cat([x_num, embs], dim=-1)
        deep_out = self.deep_mlp(x_deep)
        x_wide = torch.cat([x_num, x_ohe.float()], dim=-1)
        wide_out = self.wide_layer(x_wide)

        # sum the output from wide and deep
        out = wide_out + deep_out
        # out = torch.sigmoid(out)
        return out




