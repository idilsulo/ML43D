import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super(ThreeDEPN, self).__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.leaky = nn.LeakyReLU(0.02)
        self.relu = nn.ReLU()
        self.encoder1 = nn.Conv3d(in_channels=2, out_channels=self.num_features, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm3d(32)

        self.encoder2 = nn.Conv3d(in_channels=self.num_features, out_channels=self.num_features*2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(self.num_features * 2)
        
        self.encoder3 = nn.Conv3d(in_channels=self.num_features*2, out_channels=self.num_features*4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(self.num_features * 4)
        
        self.encoder4 = nn.Conv3d(in_channels=self.num_features*4, out_channels=self.num_features*8, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(self.num_features * 8)
        # TODO: 2 Bottleneck layers

        self.linear1 = nn.Linear(self.num_features * 8, self.num_features * 8)
        self.linear2 = nn.Linear(self.num_features * 8, self.num_features * 8)

        # TODO: 4 Decoder layers

        self.decoder1 = nn.ConvTranspose3d(in_channels=self.num_features * 2 * 8, out_channels=self.num_features * 4, kernel_size=4, stride=1, padding=0)
        self.dbn1 = nn.BatchNorm3d(self.num_features * 4)
        
        self.decoder2 = nn.ConvTranspose3d(in_channels=self.num_features * 4 * 2, out_channels=self.num_features * 2, kernel_size=4, stride=2, padding=1)
        self.dbn2 = nn.BatchNorm3d(self.num_features * 2)
        
        self.decoder3 = nn.ConvTranspose3d(in_channels=self.num_features * 2 * 2, out_channels=self.num_features, kernel_size=4, stride=2, padding=1)
        self.dbn3 = nn.BatchNorm3d(self.num_features)
        
        self.decoder4 = nn.ConvTranspose3d(in_channels=self.num_features * 2, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        # Reshape and apply bottleneck layers
        x1 = self.leaky(self.encoder1(x))
        x2 = self.leaky(self.bn2(self.encoder2(x1)))
        x3 = self.leaky(self.bn3(self.encoder3(x2)))
        x4 = self.leaky(self.bn4(self.encoder4(x3)))

        x = x4.view(b, -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x = torch.cat((x, x4), dim=1)
        x = self.relu(self.dbn1(self.decoder1(x)))
        x = torch.cat((x, x3), dim=1)
        x = self.relu(self.dbn2(self.decoder2(x)))
        x = torch.cat((x, x2), dim=1)
        x = self.relu(self.dbn3(self.decoder3(x)))
        x = torch.cat((x, x1), dim=1)
        x = self.decoder4(x)
        x = torch.squeeze(x, dim=1)
        # TODO: Log scaling
        x = torch.log(torch.abs(x) + 1)
        return x
