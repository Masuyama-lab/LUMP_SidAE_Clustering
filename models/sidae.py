import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.models import resnet18

def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def DCL(z1, z2, tau=0.5, stop_grad=False):
    batch_size = z1.shape[0]
    if stop_grad:
        z2 = z2.detach()

    # z1 = z1.reshape(z1.shape[0], -1)
    # z2 = z2.reshape(z2.shape[0], -1)
    feature = torch.cat([z1, z2])
    feature = F.normalize(feature, dim=1)

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to("cuda:0")
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to("cuda:0")
    labels = labels[~mask].view(labels.shape[0], -1)

    similarity_matrix = torch.matmul(feature, feature.T)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to("cuda:0")

    logits = logits / tau

    L = nn.CrossEntropyLoss()(logits, labels)

    return L

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.linear1 = nn.Linear(out_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class CNN5Decoder_layer(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size: int = 3,
            output_padding: int = 1,
            padding: int = 1,
            stride: int = 2,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                output_padding=output_padding,
                padding=padding,
                stride=stride,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class CNN5Decoder(nn.Module):
    def __init__(self, latent_dim=2048):
        super(CNN5Decoder, self).__init__()

        self.linear = nn.Linear(latent_dim, 512)

        self.transConv = nn.Sequential(
            # CNN5Decoder_layer(512, 512),
            CNN5Decoder_layer(512, 256),
            CNN5Decoder_layer(256, 128),
            CNN5Decoder_layer(128, 64),
            CNN5Decoder_layer(64, 64),
            CNN5Decoder_layer(64, 3)
        )

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(out.shape[0], -1, 1, 1)
        out = self.transConv(out)

        return out

class CNN6Decoder(nn.Module):
    def __init__(self, latent_dim=2048):
        super(CNN6Decoder, self).__init__()

        self.linear = nn.Linear(latent_dim, 512)

        self.transConv = nn.Sequential(
            CNN5Decoder_layer(512, 512),
            CNN5Decoder_layer(512, 256),
            CNN5Decoder_layer(256, 128),
            CNN5Decoder_layer(128, 64),
            CNN5Decoder_layer(64, 64),
            CNN5Decoder_layer(64, 3)
        )

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(out.shape[0], -1, 1, 1)
        out = self.transConv(out)

        return out

class BasicBlockDec(nn.Module):
    def __init__(self, input_channels, stride: int = 1, kernel_size: int = 3):
        super().__init__()
        planes = int(input_channels / stride)

        self.conv2 = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1:
            self.conv1 = nn.ConvTranspose2d(input_channels, planes, kernel_size=kernel_size, stride=stride, padding=1,
                                            output_padding=1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(input_channels, planes, kernel_size=3, stride=stride, padding=1, output_padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.conv1 = nn.ConvTranspose2d(input_channels, planes, kernel_size=kernel_size, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.upsample = None

    def forward(self, x):
        identity = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Resnet18Decoder(nn.Module):
    def __init__(self):
        super(Resnet18Decoder, self).__init__()
        self.layers = [2, 2, 2, 2]
        self.input_channels = 512

        self.linear = nn.Linear(2048, 512)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer4 = self._make_layer(BasicBlockDec, 256, self.layers[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, self.layers[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, self.layers[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, self.layers[0], stride=1)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []

        for stride in reversed(strides):
            layers += [block(self.input_channels, stride)]
        self.input_channels = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(out.shape[0], -1, 1, 1)
        # out = self.upsample(out)
        out = self.layer4(out)
        out = self.layer3(out)
        out = self.layer2(out)
        out = self.layer1(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.conv1(out)
        out = F.sigmoid(out)

        cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        # imagenet_norm = [[0.4802, 0.4480, 0.3975], [0.2770, 0.2691, 0.2821]]
        transform = transforms.Normalize(*cifar_norm)
        out = transform(out)

        return out

class Encoder(nn.Module):
    def __init__(self, backbone, projector):
        super(Encoder, self).__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x, siam_learn=False):
        z, z_mid = self.backbone(x, plot_metrics=True)
        z = self.projector(z)

        if siam_learn:
            return z, z_mid
        else:
            return z_mid


class SidAE(nn.Module):
    def __init__(self, backbone=resnet18(), weight=0.005):
        super().__init__()

        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )

        self.predictor = prediction_MLP()
        self.decoder = None

        self.recon_loss_func = nn.MSELoss()
        self.weight = weight
        self.use_mid_latent = False

    def set_decoder(self, decoder):
        if decoder == 'resnet18':
            self.decoder = Resnet18Decoder()
        elif decoder == 'cnn5':
            self.decoder = CNN5Decoder()
        elif decoder == "cnn6":
            self.decoder = CNN6Decoder()

    def forward(self, x1, x2, x, epoch):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)

        p1, p2 = h(z1), h(z2)
        self.siam_L = D(p1, z2) / 2 + D(p2, z1) / 2

        if self.decoder is None:
            self.L = self.siam_L
        else:
            if epoch < 90:
                z1 = z1.detach()
                z2 = z2.detach()
                recon_1 = self.decoder(z1)
                recon_2 = self.decoder(z2)

                self.recon_L = self.recon_loss_func(recon_1, x) / 2 + self.recon_loss_func(recon_2, x) / 2
                return {'siam_L': self.siam_L, 'recon_L': self.recon_L}
            else:
                recon_1 = self.decoder(self.encoder(x1))
                recon_2 = self.decoder(self.encoder(x2))

            self.recon_L = self.recon_loss_func(recon_1, x) / 2 + self.recon_loss_func(recon_2, x) / 2

            self.L = self.recon_L * self.weight + self.siam_L * (1-self.weight)

        return {'loss': self.L}


if __name__ == "__main__":
    model = SidAE()
    model = torch.nn.DataParallel(model)
    x = torch.randn((8, 3, 32, 32))
    x1 = torch.randn((8, 3, 32, 32))
    x2 = torch.randn_like(x1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in range(50):
        # model.forward(x1, x2).backward()
        optimizer.zero_grad()
        loss = model.forward(x1, x2, x)['loss']
        loss.backward()
        optimizer.step()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time

    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469
