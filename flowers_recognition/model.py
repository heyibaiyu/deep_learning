from libs import *


def accuracy(out, label):
    _, predicted = torch.max(out, 1)
    result = torch.sum(predicted == label).item() / len(predicted)
    return torch.tensor(result)

"""
10 Epoch result of SimpleModel: 
Epoch [1/10], Loss: 1.2425, Accuracy: 0.4686
Epoch [2/10], Loss: 1.1516, Accuracy: 0.5367
Epoch [3/10], Loss: 1.0136, Accuracy: 0.5996
Epoch [4/10], Loss: 0.9568, Accuracy: 0.6409
Epoch [5/10], Loss: 0.9043, Accuracy: 0.6365
Epoch [6/10], Loss: 0.9210, Accuracy: 0.6476
Epoch [7/10], Loss: 1.0094, Accuracy: 0.5974
Epoch [8/10], Loss: 1.0816, Accuracy: 0.6102
Epoch [9/10], Loss: 1.2761, Accuracy: 0.6077
Epoch [10/10], Loss: 1.6248, Accuracy: 0.5689
"""
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        dummy_input = torch.randn(1, 3, 224, 224)
        output_shape = self.conv(dummy_input).shape
        flattened_size = output_shape[1] * output_shape[2] * output_shape[3]
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # input: 128 x 3 x 224 x 224
        # out = self.conv1(x)
        # print('conv1 shape', out.shape) # 128 x 32 x 222 x 222
        # out = self.pool1(out)
        # print('pool1 shape', out.shape) # 128 x 32 x 111 x 111
        # out = self.conv2(out)
        # print('conv2 shape', out.shape) # 128 x 64 x 109 x 109
        # out = self.pool2(out)
        # print('pool2 shape', out.shape) # 128 x 64 x 54 x 54
        # out = self.conv3(out)
        # print('conv3 shape', out.shape) # 128 x 128 x 52 x 52
        # out = self.pool3(out)
        # print('pool3 shape', out.shape) # 128 x 128 x 26 x 26

        # print('input shape', x.shape)
        out = self.conv(x)
        # print('conv shape', out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        # print('fc1 shape', out.shape)
        out = self.fc2(out)
        # print('fc2 shape', out.shape)
        out = self.fc3(out)
        # print('fc3 shape', out.shape)
        return out

    def train_loss(self, batch):
        img, label = batch
        out = self.forward(img)
        loss = F.cross_entropy(out, label)
        return loss

    def val_loss(self, batch):
        img, label = batch
        out = self.forward(img)
        loss = F.cross_entropy(out, label)
        acc = accuracy(out, label)
        return loss, acc


class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        # Freeze all parameters in the model to avoid weight update during training
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer to match target problem for 5 classes
        # only the parameters of model.fc have requires_grad=True
        num_fc_in = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_fc_in, num_classes)


    def forward(self, x):
        return self.model(x)

    def train_loss(self, batch):
        img, label = batch
        out = self.forward(img)
        loss = F.cross_entropy(out, label)
        return loss

    def val_loss(self, batch):
        img, label = batch
        out = self.forward(img)
        loss = F.cross_entropy(out, label)
        acc = accuracy(out, label)
        return loss, acc