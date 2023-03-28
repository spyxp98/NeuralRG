# %%
"""
Based on paper: 
[1] Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio "Density estimation using Real NVP"
arXiv:1605.08803v3 [cs.LG] 
and 
https://github.com/karpathy/pytorch-normalizing-flows
"""

# %%
import torch
import torch.nn as nn
from torchvision.ops import MLP


class AffineCouplingLayer(nn.Module):
    """
    Implements affine coupling layer as described in [1]
    Layer: D -> D
    Parameters:
      D: int - full dimensioanlity of layer
      d: int - intermediate dimensionality inside of layer
      nn_class: nn.Module - class of neural networks used inside of layer
      hidden_channels: list[int] - hidden_channels in nn_class  
      scale: bool - whether to scale or not 
      shift: bool - whether to shift or not
    if scale = shift = False, then layer implements identity transformation
    """

    def __init__(self, D, d=None, nn_class=MLP, hidden_channels=[], scale=True, shift=True) -> None:
        super().__init__()
        self.D = D
        self.d = D // 2 if d == None else d
        self.nn_class = nn_class
        self.nn_hidden_channels_ = hidden_channels
        self.nn_hidden_channels_.append(self.D - self.d)

        self.s = lambda x: x.new_zeros(x.size(0), self.D - self.d)
        self.t = lambda x: x.new_zeros(x.size(0), self.D - self.d)
        if scale:
            self.s = self.nn_class(
                in_channels=self.d,
                hidden_channels=self.nn_hidden_channels_,
                activation_layer=nn.ReLU,
                bias=True,
                dropout=0
            )
        if shift:
            self.t = self.nn_class(
                in_channels=self.d,
                hidden_channels=self.nn_hidden_channels_,
                activation_layer=nn.ReLU,
                bias=True,
                dropout=0
            )

    """
  x -> z:
  z[:d] = x[:d]
  z[d:D] = x[d:D] * exp(s(x[:d])) + t(x[:d])
  """

    def forward(self, x):
        x0, x1 = x[:, 0:self.d], x[:, self.d:self.D]
        s = self.s(x0)
        t = self.t(x0)
        z0 = x0
        z1 = torch.exp(s) * x1 + t
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    """
  x0 = z0
  x1 = (z1 - t) * exp(-s)
  """

    def backward(self, z):
        z0, z1 = z[:, 0:self.d], z[:, self.d:self.D]
        s = self.s(z0)
        t = self.t(z0)
        x0 = z0
        x1 = (z1 - t) * torch.exp(-s)
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class RNVP(nn.Module):
    def __init__(self, D, d=None, nn_class=MLP, hidden_channels=[], num_of_layers=2) -> None:
        super().__init__()
        self.flows = nn.ModuleList([
            AffineCouplingLayer(D, d, nn_class, hidden_channels=hidden_channels) for _ in range(num_of_layers)
        ])

    def forward(self, x):
        log_det = torch.zeros(x.shape[0])
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        log_det = torch.zeros(z.shape[0])
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

# %%


class NormalizingFlowModel(nn.Module):
    """
    Pair of (prior distribution, flow)
    """

    def __init__(self, prior, D, d=None, nn_class=MLP, hidden_channels=[], num_of_layers=2) -> None:
        super().__init__()
        self.prior = prior
        self.flow = RNVP(D, d, nn_class, hidden_channels, num_of_layers)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        with torch.no_grad:
            self.flow.eval()
            z = self.prior.sample((num_samples, ))
            xs, _ = self.flow.backward(z)
        self.flow.train()
        return xs
