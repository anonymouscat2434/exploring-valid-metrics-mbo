from typing import Tuple
import torch
from torch import nn
from torch.nn.utils import spectral_norm as spec_norm
from torch.nn import functional as F
import numpy as np

from torch.distributions import Distribution
from torch.distributions import Laplace, Normal, Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

from ...setup_logger import get_logger
logger = get_logger(__name__)

from ..positional_embedding import TimestepEmbedding

def get_linear(n_in, n_out):
    layer = nn.Linear(n_in, n_out)
    nn.init.xavier_uniform(layer.weight.data, 1.0)
    return layer


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        n_hidden: int,
        n_out: int,
        n_layers: int = 4,
        tau: float = 1.0,
        pos_embedding: bool = True,
        norm_layer: str = "layer_norm",
        spec_norm: bool = False,
        discrete: bool = True,
        discrete_dims: Tuple = None,
        one_hot_forward: bool = True,
    ):
        super().__init__()
        if spec_norm:
            logger.info("G: spectral norm enabled")
            sn_fn = spec_norm
        else:
            sn_fn = lambda x: x

        if norm_layer == "layer_norm":
            layer_norm_fn = nn.LayerNorm
        elif norm_layer == "batch_norm":
            layer_norm_fn = nn.BatchNorm1d
        elif norm_layer is None:
            layer_norm_fn = nn.Identity
        else:
            raise NotImplementedError("{} unknown".format(norm_layer))

        encoder = []
        for j in range(n_layers+1):
            if j == 0:
                this_in, this_out = n_out+n_hidden, n_hidden
            elif j == (n_layers):
                this_in, this_out = n_hidden*2, z_dim*2
            else:
                this_in, this_out = n_hidden*2, n_hidden
            encoder.append(nn.Sequential(
                sn_fn(nn.Linear(this_in, this_out)),
                layer_norm_fn(this_out),
                #nn.BatchNorm1d(this_out) if use_norm else nn.Identity(),
                nn.ReLU() if j != n_layers else nn.Identity()
            ))
        self.encoder = nn.ModuleList(encoder)

        #self.embed = nn.Linear(1, n_hidden)
        #self.embed = nn.Identity()
        if pos_embedding:
            self.embed = TimestepEmbedding(
                embedding_dim=n_hidden,
                hidden_dim=n_hidden,
                output_dim=n_hidden
            )
        else:
            self.embed = nn.Linear(1, n_hidden)

        decoder = []
        for j in range(n_layers+1):
            if j == 0:
                this_in, this_out = z_dim+n_hidden, n_hidden
            elif j == (n_layers):
                this_in, this_out = n_hidden*2, n_out
            else:
                this_in, this_out = n_hidden*2, n_hidden
            decoder.append(nn.Sequential(
                sn_fn(nn.Linear(this_in, this_out)),
                #nn.BatchNorm1d(this_out) if use_norm else nn.Identity(),
                layer_norm_fn(this_out),
                nn.ReLU() if j != n_layers else nn.Identity()
            ))
        self.decoder = nn.ModuleList(decoder)

        self.tau = tau
        self.discrete = discrete
        self.discrete_dims = discrete_dims
        self.one_hot_forward = one_hot_forward

        self.z_dim = z_dim

    def encode(self, x, y) -> Distribution:
        h = x
        yy = self.embed(y)
        for mod in self.encoder:
            h = mod( torch.cat((h, yy), dim=1) )
        mu, logsigma = torch.chunk(h, chunks=2, dim=1)
        qdistn = Normal(mu, torch.exp(logsigma))
        return qdistn

    def decode(self, z, y) -> Distribution:
        
        # TODO: also support discrete units
        h = z
        yy = self.embed(y)
        for mod in self.decoder:
            h = mod( torch.cat((h, yy), dim=1) )
        if self.discrete:
            logits_rs = h.view(-1, *self.discrete_dims)  # (bs, 8, 4)
            p_x_given_zy = RelaxedOneHotCategorical(
                logits=logits_rs, 
                temperature=self.tau
            )
        else:
            p_x_given_zy = Laplace(h, torch.ones_like(h))
        
        return p_x_given_zy

    def forward(self, x, y):
        this_z = self.encode(x).rsample()
        return self.decode(this_z, y)

    def sample_z(self, bs):
        return torch.zeros((bs, self.z_dim)).normal_(0, 1)

    @torch.no_grad()
    def sample(self, y_cond, z=None):
        self.eval()
        if z is None:
            z = self.sample_z(y_cond.size(0)).to(y_cond.device)
        p_x_given_zy = self.decode(z, y_cond)
        if self.discrete:
            # RelaxedOneHotCategorical does not allow a one-hot
            # pass, so do it with gumbel_softmax in Functional
            samples_onehot = F.gumbel_softmax(
                p_x_given_zy.logits, 
                tau=self.tau, 
                dim=-1, 
                hard=True
            )  # (bs, 8, 4) one-hot
            return samples_onehot.reshape(-1, 
                                          np.prod(self.discrete_dims))
        else:
            # Use sd=1 for likelihood but do not actually sample
            # with rsample(), it's gonna give us wildly crazy samples 
            # and I don't  want to tune or learn the sd at this point.
            # Just return the mean of this distribution with `loc`.
            return p_x_given_zy.loc

if __name__ == '__main__':
    EncoderDecoder(z_dim=32, n_hidden=256, n_out=16, discrete=False)