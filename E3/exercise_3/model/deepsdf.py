import torch.nn as nn
import torch

class WeightedLinear(nn.Module):
    def __init__(self, x_in, x_out, dropout_prob=0.2):
        super().__init__()
        self.weighted = torch.nn.utils.weight_norm(nn.Linear(x_in, x_out))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)


    def forward(self, x):
        x = self.dropout(self.relu(self.weighted(x)))
        return x


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.w1 = WeightedLinear(latent_size+3, 512, dropout_prob=dropout_prob)
        self.w2 = WeightedLinear(512, 512, dropout_prob=dropout_prob)
        self.w3 = WeightedLinear(512, 512, dropout_prob=dropout_prob)
        self.w4 = WeightedLinear(512, 253, dropout_prob=dropout_prob)

        self.w5 = WeightedLinear(512, 512, dropout_prob=dropout_prob)
        self.w6 = WeightedLinear(512, 512, dropout_prob=dropout_prob)
        self.w7 = WeightedLinear(512, 512, dropout_prob=dropout_prob)
        self.w8 = WeightedLinear(512, 512, dropout_prob=dropout_prob)

        self.linear = torch.nn.Linear(512, 1)



    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass

        x = self.w1(x_in)
        x = self.w2(x)
        x = self.w3(x)
        x = self.w4(x)

        x = torch.cat((x, x_in), dim=1)

        x = self.w5(x)
        x = self.w6(x)
        x = self.w7(x)
        x = self.w8(x)

        x = self.linear(x)

        return x
