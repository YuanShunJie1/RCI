import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class PadAndSubsample(nn.Module):
    def __init__(self, planes):
        super(PadAndSubsample, self).__init__()
        self.planes = planes

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = PadAndSubsample(planes)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)



def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

class MLPNet(nn.Module):
    def __init__(self, emb_length, num_passive):
        super(MLPNet, self).__init__()
        self._size = int(28//num_passive)
        
        self.fc1 = nn.Linear(self._size * 28, 256)
        self.fc2 = nn.Linear(256, emb_length)

    def forward(self, x):
        x = x.reshape(-1, self._size * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Yeast
class MLPNet1(nn.Module):
    def __init__(self, emb_length, num_passive):
        super(MLPNet1, self).__init__()
        self._size = int(8//num_passive)
        self.fc1 = nn.Linear(self._size, 256)
        self.fc2 = nn.Linear(256, emb_length)

    def forward(self, x):
        x = x.reshape(-1, self._size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Letter
class MLPNet2(nn.Module):
    def __init__(self, emb_length, num_passive):
        super(MLPNet2, self).__init__()
        self._size = int(16//num_passive)
        self.fc1 = nn.Linear(self._size, 256)
        self.fc2 = nn.Linear(256, emb_length)

    def forward(self, x):
        x = x.reshape(-1, self._size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BottomModelForMnist(nn.Module):
    def __init__(self, emb_length, num_passive):
        super(BottomModelForMnist, self).__init__()
        self.mlpnet = MLPNet(emb_length, num_passive)
    def forward(self, x):
        x = self.mlpnet(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForMnist(nn.Module):
    def __init__(self, emb_length):
        super(TopModelForMnist, self).__init__()
        self.fc1top = nn.Linear(emb_length, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(emb_length)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, concated_embeddings):
        x = concated_embeddings
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return x


class Network_Mnist(nn.Module):
    def __init__(self, num_passive, emb_length):
        super(Network_Mnist, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForMnist(emb_length, num_passive) for i in range(num_passive)
        ])
        self.active = TopModelForMnist(emb_length*num_passive)

    def forward(self, concated_embeddings):
        logit = self.active(concated_embeddings)
        return logit

    def _aggregate(self, embeddings):
        concated_embeddings = torch.cat(embeddings, dim=1)
        return concated_embeddings



class BottomModelForYeast(nn.Module):
    def __init__(self, emb_length, num_passive):
        super(BottomModelForYeast, self).__init__()
        self.mlpnet = MLPNet1(emb_length,num_passive)

    def forward(self, x):
        x = self.mlpnet(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForYeast(nn.Module):
    def __init__(self, emb_length):
        super(TopModelForYeast, self).__init__()
        self.fc1top = nn.Linear(emb_length, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(emb_length)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, concated_embeddings):
        x = concated_embeddings
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return x

class Network_Yeast(nn.Module):
    def __init__(self, num_passive, emb_length):
        super(Network_Yeast, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForYeast(emb_length, num_passive) for i in range(num_passive)
        ])
        self.active = TopModelForYeast(emb_length*num_passive)

    def forward(self, concated_embeddings):
        logit = self.active(concated_embeddings)
        return logit

    def _aggregate(self, embeddings):
        concated_embeddings = torch.cat(embeddings, dim=1)
        return concated_embeddings



class BottomModelForLetter(nn.Module):
    def __init__(self, emb_length, num_passive):
        super(BottomModelForLetter, self).__init__()
        self.mlpnet = MLPNet2(emb_length,num_passive)

    def forward(self, x):
        x = self.mlpnet(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForLetter(nn.Module):
    def __init__(self, emb_length):
        super(TopModelForLetter, self).__init__()
        self.fc1top = nn.Linear(emb_length, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 26)
        self.bn0top = nn.BatchNorm1d(emb_length)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, concated_embeddings):
        x = concated_embeddings
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return x

class Network_Letter(nn.Module):
    def __init__(self, num_passive, emb_length):
        super(Network_Letter, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForLetter(emb_length, num_passive) for i in range(num_passive)
        ])
        self.active = TopModelForLetter(emb_length*num_passive)

    def forward(self, concated_embeddings):
        logit = self.active(concated_embeddings)
        return logit

    def _aggregate(self, embeddings):
        concated_embeddings = torch.cat(embeddings, dim=1)
        return concated_embeddings



class BottomModelForCifar10(nn.Module):
    def __init__(self, emb_length):
        super(BottomModelForCifar10, self).__init__()
        self.resnet18 = resnet18(num_classes=emb_length)

    def forward(self, x):
        x = self.resnet18(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForCifar10(nn.Module):
    def __init__(self, emb_length):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(emb_length, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(emb_length)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self, concated_embeddings):
        x = concated_embeddings
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return x

class Network_Cifar10(nn.Module):
    def __init__(self, num_passive, emb_length):
        super(Network_Cifar10, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForCifar10(emb_length) for i in range(num_passive)
        ])
        self.active = TopModelForCifar10(num_passive*emb_length)

    def forward(self, concated_embeddings):
        logit = self.active(concated_embeddings)
        return logit

    def _aggregate(self, embeddings):
        concated_embeddings = torch.cat(embeddings, dim=1)
        return concated_embeddings


class BottomModelForCifar100(nn.Module):
    def __init__(self, emb_length):
        super(BottomModelForCifar100, self).__init__()
        # self.resnet20 = resnet20(num_classes=emb_length)
        self.resnet18 = resnet18(num_classes=emb_length)

    def forward(self, x):
        x = self.resnet18(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForCifar100(nn.Module):
    def __init__(self, emb_length):
        super(TopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(emb_length, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 100)
        self.bn0top = nn.BatchNorm1d(emb_length)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self, concated_embeddings):
        x = concated_embeddings
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return x


class Network_Cifar100(nn.Module):
    def __init__(self, num_passive, emb_length):
        super(Network_Cifar100, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForCifar100(emb_length) for i in range(num_passive)
        ])
        self.active = TopModelForCifar100(num_passive*emb_length)

    def forward(self, concated_embeddings):
        logit = self.active(concated_embeddings)
        return logit

    def _aggregate(self, embeddings):
        concated_embeddings = torch.cat(embeddings, dim=1)
        return concated_embeddings

class BasicBlock_L(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_L, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet_M(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_M, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_hidden=False, return_activation=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        activation1 = out
        out = self.layer2(out)
        activation2 = out
        out = self.layer3(out)
        activation3 = out
        out = self.layer4(out)
        
        out = self.avgpool(out)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)

        if return_hidden:
            return out, hidden
        elif return_activation:  # for NAD
            return out, activation1, activation2, activation3
        else:
            return out

def resnet18(num_classes=12):
    return ResNet_M(BasicBlock_L, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=12):
    return ResNet_M(BasicBlock_L, [3, 4, 6, 3], num_classes=num_classes)

class BottomModelForImageNet12(nn.Module):
    def __init__(self,emb_length):
        super(BottomModelForImageNet12, self).__init__()
        self.resnet34 = resnet34(num_classes=emb_length)

    def forward(self, x):
        x = self.resnet34(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForImageNet12(nn.Module):
    def __init__(self,input_dims):
        super(TopModelForImageNet12, self).__init__()
        self.fc1top = nn.Linear(input_dims, 512)
        self.fc2top = nn.Linear(512, 128)
        self.fc3top = nn.Linear(128, 12)
        self.bn0top = nn.BatchNorm1d(input_dims)
        self.bn1top = nn.BatchNorm1d(512)
        self.bn2top = nn.BatchNorm1d(128)
        self.apply(weights_init)

    def forward(self, concated_embeddings):
        x = concated_embeddings
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return x


class Network_ImageNet12(nn.Module):
    def __init__(self, num_passive, emb_length):
        super(Network_ImageNet12, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForImageNet12(emb_length) for i in range(num_passive)
        ])
        self.active = TopModelForImageNet12(num_passive*emb_length)

    def forward(self, concated_embeddings):
        logit = self.active(concated_embeddings)
        return logit

    def _aggregate(self, embeddings):
        concated_embeddings = torch.cat(embeddings, dim=1)
        return concated_embeddings


# "mnist",
# "fashionmnist",
# "cifar10",
# "cifar100",
# "yeast",
# "letter",
# "imagenet12"


entire = {
    'mnist': Network_Mnist,
    'fashionmnist': Network_Mnist,
    'cifar10':Network_Cifar10,
    'cifar100':Network_Cifar100,
    'yeast':Network_Yeast,
    'letter':Network_Letter,
    'imagenet12':Network_ImageNet12,
}
