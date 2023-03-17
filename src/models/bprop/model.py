import os
from typing import Dict, Union
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
# from torch.cuda.amp import autocast, GradScaler
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from ... import utils as ut

from ...setup_logger import get_logger
logger = get_logger(__name__)

from ..base_model import BaseModel

class TensorWrapper(nn.Module):
    def __init__(self, tensor, classifier):
        super().__init__()
        self.tensor = nn.Parameter(tensor, requires_grad=True)
        self.classifier = classifier
    def forward(self):
        return self.classifier.predict(self.tensor)
    def extra_repr(self):
        return "tensor={}".format(self.tensor.shape)

class Bprop(BaseModel):

    DEFAULT_ARGS = {}

    def _validate_args(self, dd):
        pass

    def __init__(
        self,
        classifier,
        n_iters=1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.classifier = classifier
        self.n_iters = n_iters

    def set_classifier(self, classifier):
        self.classifier = classifier

    def sample(self, y, z=None, **kwargs):
        if z is not None:
            raise ValueError("This class does not support conditioning on a custom z")

        # We want to backprop directly on this
        xs = torch.randn((y.size(0), self.n_out))
        xs = xs.to(self.rank)

        xs_on_cls = TensorWrapper(xs, self.classifier)
        
        opt = Adam( xs_on_cls.parameters(), lr=0.01)
        #logger.debug("Optimiser for bprop on xs: {}".format(opt))

        losses = []
        for iter_ in range(self.n_iters):
            opt.zero_grad()
            preds = xs_on_cls()
            pred_loss = torch.mean((preds-y)**2)
            pred_loss.backward()
            opt.step()
            losses.append(pred_loss.item())

        return xs_on_cls.tensor.data.detach()

    @torch.no_grad()
    def sample_z(self, *args, **kwargs):
        return self.gen.sample_z(*args, **kwargs)

    def train(self):
        pass

    def eval(self):
        pass

    def _run_on_batch(self, batch, train=True, savedir=None, classes=None, **kwargs):
        raise NotImplementedError("This class does not need to be trained")

    def get_state_dict(self):
        pass

    def set_state_dict(self, state_dict, load_opt=True, strict=True):
        pass
