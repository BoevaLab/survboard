from torch import nn

ACTIVATION_FN_FACTORY = {
    "relu": nn.ReLU(),
    "prelu": nn.PReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
    "lrelu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "softmax": nn.Softmax(dim=1),
}
