import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F

from ...setup_logger import get_logger

logger = get_logger(__name__)


def get_linear(n_in, n_out):
    layer = nn.Linear(n_in, n_out)
    nn.init.xavier_uniform(layer.weight.data, 1.0)
    return layer


class Generator(nn.Module):
    def __init__(
        self,
        z_dim,
        n_hidden,
        n_out,
        n_layers=4,
        tau=1.0,
        use_norm=True,
        spec_norm=False,
        discrete=True,
        discrete_dims=None,
        one_hot_forward=True,
    ):
        """_summary_

        Args:
            z_dim (int): dimension of z
            n_hidden (int): number of hidden units.
            n_out (int): output dimension. If generated inputs are discrete,
                then this is the number of discrete  variables multiplied by
                their possible values (e.g. 8*4 for TFBind8 dataset).
            n_layers (int, optional): Number of hidden layers. Defaults to 4.
            tau (float, optional): _description_. Defaults to 1.0.
            discrete (bool, optional): Is this a discrete generation task? If
                true, then Gumbel-Softmax will be used to sample a discrete
                variable. Defaults to True.
            discrete_dims (_type_, optional): the discrete dimensions for
                Gumbel Softmax. For instance, for TFBind8 there are 8 categorical
                variables each taking on one of four values, so the output logits
                of the generator will be of shape (bs, 8*4). That means
                `discrete_dims` needs to be (8,4) so that we can reshape the
                tensor into (bs,8,4) and compute the softmax over the last
                dimension.
            one_hot_forward (bool, optional): Do we use straight-thru estimator
              for the discrete generation? Defaults to True.
        """
        
        super().__init__()
        if spec_norm:
            logger.info("G: spectral norm enabled")
            sn_fn = spectral_norm
        else:
            sn_fn = lambda x: x

        self.z_dim = z_dim
        self.pre = nn.Sequential(
            sn_fn(nn.Linear(z_dim + n_hidden, n_hidden)),
            nn.LayerNorm(n_hidden) if use_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.embed = nn.Linear(1, n_hidden)
        hidden = []
        # TODO: embed y with n_hidden units
        for _ in range(n_layers):
            hidden.append(
                nn.Sequential(
                    sn_fn(nn.Linear(n_hidden * 2, n_hidden)),
                    nn.LayerNorm(n_hidden) if use_norm else nn.Identity(),
                    nn.LeakyReLU(0.2),
                )
            )
        self.hidden = nn.ModuleList(hidden)
        self.post = sn_fn(nn.Linear(n_hidden, n_out))
        self.tau = tau
        self.discrete = discrete
        self.discrete_dims = discrete_dims
        self.one_hot_forward = one_hot_forward

    def forward(self, z, y):
        if self.training:
            onehot_out = self.one_hot_forward
        else:
            # At inference time, sample one-hot, not continuous
            onehot_out = True
        y_embed = self.embed(y)
        h = self.pre(torch.cat((z, y_embed), dim=1))
        for j in range(len(self.hidden)):
            h = self.hidden[j](torch.cat((h, y_embed), dim=1))
        if self.discrete:
            logits = self.post(h)
            logits_rs = logits.view(-1, *self.discrete_dims)  # (bs, 8, 4)
            probs_rs = F.gumbel_softmax(
                logits_rs, tau=self.tau, dim=-1, hard=onehot_out
            )  # (bs, 8, 4) one-hot
            # TODO: change tau at inference time???
            # probs_rs = F.softmax(logits_rs, dim=2)
            probs = probs_rs.view(probs_rs.size(0), -1)  # (bs, 8*4) flattened
            return probs
        else:
            return self.post(h)

    def sample_z(self, bs):
        return torch.zeros((bs, self.z_dim)).normal_(0, 1)

    @torch.no_grad()
    def sample(self, y_cond, z=None):
        if z is None:
            z = self.sample_z(y_cond.size(0)).to(y_cond.device)
        xfake = self.forward(z, y_cond)
        return xfake


class Discriminator(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, z_dim):
        super().__init__()

        spec_norm = spectral_norm

        self.pre = nn.Sequential(
            spec_norm(get_linear(n_in, n_hidden)),
            # nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )
        hidden = []
        for _ in range(n_layers):
            hidden.append(
                nn.Sequential(
                    spec_norm(get_linear(n_hidden * 2, n_hidden)),
                    # nn.LayerNorm(n_hidden),
                    nn.LeakyReLU(0.2),
                )
            )
        self.hidden = nn.ModuleList(hidden)
        self.pred_z = get_linear(n_hidden, z_dim)
        self.pred_y = get_linear(n_hidden, 1)
        self.y_proj = spec_norm(get_linear(1, n_hidden))
        # self.embed = nn.Linear(1, n_hidden)
        self.embed = spec_norm(get_linear(1, n_hidden))
        self.phi = spec_norm(get_linear(n_hidden, 1))

    def forward(self, x, y):
        y_proj = self.embed(y)
        h = self.pre(x)
        for j in range(len(self.hidden)):
            h = self.hidden[j](torch.cat((h, y_proj), dim=1))
        rf = self.phi(h)  # + torch.sum(y_proj*h, dim=1, keepdim=True)
        return rf, self.pred_z(h), self.pred_y(h)
