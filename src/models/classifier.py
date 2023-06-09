import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from haven import haven_utils as hu

#from .resnet18 import Resnet18

from torch.optim import AdamW as Adam

from ..setup_logger import get_logger
logger = get_logger(__name__)

#from ..models import utils as ut

class MLPClassifier(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            #nn.BatchNorm1d(n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )
        hidden = []
        for _ in range(n_layers):
            hidden.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.LayerNorm(n_hidden),
                #nn.BatchNorm1d(n_hidden),
                nn.LeakyReLU(0.2)
            ))
        self.hidden = nn.ModuleList(hidden)
        self.out = nn.Linear(n_hidden, 1)
        self.n_in = n_in
    
    def features(self, x, all_features=True, **kwargs):
        buf = []
        h = self.pre(x)
        for j in range(len(self.hidden)):
            h = self.hidden[j](h)
            buf.append(h)
        if all_features:
            return torch.cat(buf, dim=1)
        else:
            return buf[-1]

    def forward(self, x):
        h = self.pre(x)
        for j in range(len(self.hidden)):
            h = self.hidden[j](h)
        pred_y = self.out(h)
        return pred_y

# model definition
class Classifier:
    def __init__(
        self,
        model_kwargs,
        optim_kwargs,
        rank=0,
        verbose=False,
    ):

        self.model = MLPClassifier(**model_kwargs)
        self.model.to(rank)

        self.rank = rank

        logger.info("model: {}".format(self.model))

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        beta1 = optim_kwargs.pop('beta1')
        beta2 = optim_kwargs.pop('beta2')
        self.opt = Adam(params, betas=(beta1, beta2), 
                        **optim_kwargs)

        logger.info("optim: {}".format(self.opt))

    def set_state_dict(self, state_dict, load_opt=True):
        self.model.load_state_dict(state_dict["model"])
        if load_opt:
            self.opt.load_state_dict(state_dict["opt"])

    def get_state_dict(self):
        state_dict = {"model": self.model.state_dict(), "opt": self.opt.state_dict()}
        return state_dict

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def features(self, x, **kwargs):
        self.eval()
        return self.model.features(x, **kwargs)

    def predict(self, x):
        self.eval()
        return self.model(x)

    # train the model
    def train_on_loader(
        self,
        loader,
        savedir=None,
    ):

        self.train()

        losses = []

        for _, batch in enumerate(tqdm(loader, desc="Training")):

            self.opt.zero_grad()

            x_batch, y_batch = batch
            x_batch = x_batch.to(self.rank)
            y_batch = y_batch.to(self.rank)

            y_pred = self.model(x_batch)
            loss = torch.mean( (y_pred-y_batch)**2 )

            loss.backward()
            self.opt.step()

            losses.append(loss.item())
            #accs.append(acc.item())

        return {"loss": np.mean(losses)}

    # evaluate the model
    @torch.no_grad()
    def val_on_loader(self, loader, preprocessor=None, desc="Validating"):

        self.eval()

        losses = []

        for _, batch in enumerate(tqdm(loader, desc=desc)):

            x_batch, y_batch = batch
            x_batch = x_batch.to(self.rank)
            y_batch = y_batch.to(self.rank)

            y_pred = self.model(x_batch)
            loss = torch.mean( (y_pred-y_batch)**2 )

            losses.append(loss.item())

        return {"loss": np.mean(losses)}