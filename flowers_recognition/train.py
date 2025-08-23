from libs import *
from model import SimpleModel
from utils import *


def batch_evaluate(model, val_loader):
    model.eval()
    outputs = [model.val_loss(batch) for batch in val_loader]
    batch_loss = [x[0] for x in outputs]
    batch_acc = [x[1] for x in outputs]
    loss_mean = torch.mean(torch.stack(batch_loss))
    acc_mean = torch.mean(torch.stack(batch_acc))
    return {'val_loss': loss_mean.item(), 'val_acc': acc_mean.item()}

def fit(epochs: int,
        learning_rate: float,
        model: SimpleModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        opt_fun=torch.optim.Adam):
    print('--------- start training ---------')
    history = []
    optimizer = opt_fun(model.parameters(), lr=learning_rate)
    train_losses = []
    for epoch in range(epochs):
        # training stage
        model.train()
        for batch in train_loader:
            loss = model.train_loss(batch)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validation stage
        result = batch_evaluate(model, val_loader)
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, epochs, result['val_loss'], result['val_acc']))
        history.append(result)
    return history

def train():
    path = '/Users/jing/.cache/kagglehub/datasets/alxmamaev/flowers-recognition/versions/2/flowers'
    dataset = load_data(path) # 4317 images

    # random split dataset into train and val
    # make sure the split is reproducible
    random_seed = 42
    torch.manual_seed(random_seed)
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print('--------- Train dataset size: {} ---------'.format(len(train_ds)))

    # create dataloader
    batch_size = 128
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    # show_batch(train_dl)

    # build model
    model = SimpleModel()

    num_epochs = 10
    lr = 1e-3
    opt_func = torch.optim.Adam
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


train()