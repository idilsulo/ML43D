import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        # TODO Add layers: Convolutional k->64, 64->128, 128->1024 with corresponding batch norms and ReLU
        self.conv1 = nn.Conv1d(in_channels=k, out_channels=64, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.max_pool = lambda x: torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        # TODO Add layers: Linear 1024->512, 512->256, 256->k^2 with corresponding batch norms and ReLU
        self.ln1 = nn.Linear(1024, 512)
        self.lnbn1 = nn.BatchNorm1d(512)
        self.ln2 = nn.Linear(512, 256)
        self.lnbn2 = nn.BatchNorm1d(256)
        self.ln3 = nn.Linear(256, k**2)

        self.relu = nn.ReLU()

        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k

    def forward(self, x):
        b = x.shape[0]

        # TODO Pass input through layers, applying the same max operation as in PointNetEncoder
        # TODO No batch norm and relu after the last Linear layer
        
        x = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.ln1(x)
        x = self.lnbn1(x)
        x = self.relu(x)

        x = self.ln2(x)
        x = self.lnbn2(x)
        x = self.relu(x)

        x = self.ln3(x)

        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(b, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, return_point_features=False):
        super(PointNetEncoder, self).__init__()

        # TODO Define convolution layers, batch norm layers, and ReLU
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.relu = nn.ReLU()

        self.input_transform_net = TNet(k=3)
        self.feature_transform_net = TNet(k=64)

        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]

        input_transform = self.input_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)

        # TODO: First layer: 3->64
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        feature_transform = self.feature_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        point_features = x

        # TODO: Layers 2 and 3: 64->128, 128->1024
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # This is the symmetric max operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


class PointNetClassification(nn.Module):
    def __init__(self, num_classes):
        super(PointNetClassification, self).__init__()
        self.encoder = PointNetEncoder(return_point_features=False)
        # TODO Add Linear layers, batch norms, dropout with p=0.3, and ReLU
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        # Batch Norms and ReLUs are used after all but the last layer
        # Dropout is used only directly after the second Linear layer
        # The last Linear layer reduces the number of feature channels to num_classes (=k in the architecture visualization)

    def forward(self, x):
        x = self.encoder(x)
        # TODO Pass output of encoder through your layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(return_point_features=True)
        
        # TODO: Define convolutions, batch norms, and ReLU
        self.conv1 = nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=num_classes, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        # TODO: Pass x through all layers, no batch norm or ReLU after the last conv layer

        batch_size = x.shape[0]
        n_points = x.shape[2]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()

        x = F.log_softmax(x.view(-1, self.num_classes), dim=-1)
        x = x.view(batch_size, n_points, self.num_classes)
        return x
