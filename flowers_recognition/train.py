from numpy.f2py.auxfuncs import throw_error

from libs import *
from model import *
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
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4)
    train_losses = []
    start_time = datetime.now()
    print(start_time)
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
        scheduler.step(result.get('val_acc', None))
        current_lr = optimizer.param_groups[0]['lr']

        result_str = 'Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, LR: {: .10f}'.format(epoch + 1, epochs, result['val_loss'], result['val_acc'], current_lr)
        print(result_str)
        history.append(result_str)
        print(datetime.now())

    print('Training duration: {}'.format(datetime.now() - start_time))

    return history


def train(args):
    # load kaggle data
    path = '~/.cache/kagglehub/datasets/alxmamaev/flowers-recognition/versions/2/flowers'
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
    batch_size = args.batch_size
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    # show_batch(train_dl)

    # build model
    print('--------- Build model ---------')
    if args.model_type == 'simple_cnn':
        print('SimpleCNN model')
        model = SimpleModel()
        print(model)
    elif args.model_type == 'resnet':
        print('ResNet model')
        model = ResNetTransfer(len(dataset.classes))
        print(model)
    else:
        raise ValueError("Model type {} not supported.".format(args.model_type))

    num_epochs = args.num_epochs
    lr = args.lr
    opt_func = torch.optim.Adam

    start_time = datetime.now()
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    # history = ['epoch 1: loss1, acc1', 'epoch 2: loss2, acc2', 'epoch 3: loss3, acc3']
    duration = datetime.now() - start_time


    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = args.model_type + '_' + datetime_str + '.txt'
    record_results(args, history, file_name, duration)


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--model_type', type=str, default='resnet',
                        help='Model type for training.')
    args = parser.parse_args()

    train(args)

