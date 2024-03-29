from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP, self).__init__()

        fc_dims = list(fc_dims)
        # print(fc_dims)
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            if dim == 1:
                layers.append(nn.Sigmoid())

            input_dim = dim
        # layers.append(nn.Sigmoid())
        layers.append(nn.LayerNorm(fc_dims[-1]))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        # print("mlp shape of input: ", input.shape)
        output = self.fc_layers(input)
        # print("mlp shape of output: ", output.shape)
        return output


