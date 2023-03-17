import os
from typing import Dict, Union
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
# from torch.cuda.amp import autocast, GradScaler
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW as Adam

from collections import OrderedDict

from ... import utils as ut

from ...setup_logger import get_logger
logger = get_logger(__name__)

from ..base_model import BaseModel
from .networks import Generator, Discriminator

class GAN(BaseModel):

    DEFAULT_ARGS = {}

    def _validate_args(self, dd):
        pass

    def __init__(
        self,
        gamma: float,
        update_g_every: int,
        gen_kwargs: Dict,
        disc_kwargs: Dict,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.gen = Generator(n_out=self.n_out, **gen_kwargs)
        if "n_hidden" not in disc_kwargs:
            disc_kwargs["n_hidden"] = gen_kwargs["n_hidden"]
        self.disc = Discriminator(
            n_in=self.n_out,
            **disc_kwargs,
            z_dim=gen_kwargs["z_dim"]
        )

        self.gen.to(self.rank)
        self.disc.to(self.rank)

        if self.use_ema:
            self.gen_ema = Generator(n_out=self.n_out, **gen_kwargs)
            self.gen_ema.to(self.rank)
        else:
            self.gen_ema = None

        if self.verbose and self.rank == 0:
            logger.info("gen: {}".format(self.gen))
            logger.info("# gen params: {}".format(ut.count_params(self.gen)))
            logger.info("disc: {}".format(self.disc))
            logger.info("# disc params: {}".format(ut.count_params(self.disc)))

        self.opt_g = Adam(self.gen.parameters(), **self.optim_kwargs)
        self.opt_d = Adam(self.disc.parameters(), **self.optim_kwargs)

        self.gamma = gamma
        self.update_g_every = update_g_every

    @property
    def ema_src_network(self):
        return self.gen

    @property
    def ema_tgt_network(self):
        return self.gen_ema

    @torch.no_grad()
    def sample(self, y, z=None, use_ema=False, **kwargs):
        if use_ema:
            if not self.use_ema:
                raise Exception("use_ema was set but this model has no EMA weights")
            gen = self.gen_ema
        else:
            gen = self.gen
        return gen.sample(y, z, **kwargs)

    @torch.no_grad()
    def sample_z(self, *args, **kwargs):
        return self.gen.sample_z(*args, **kwargs)

    def train(self):
        self.gen.train()
        if self.gen_ema is not None:
            self.gen_ema.train()
        self.disc.train()

    def eval(self):
        self.gen.eval()
        if self.gen_ema is not None:
            self.gen_ema.eval()
        self.disc.eval()

    def _run_on_batch(self, batch, train=True, savedir=None, classes=None, **kwargs):

        if train:
            self.train()
        else:
            self.eval()

        if train:
            self.opt_g.zero_grad()
            self.opt_d.zero_grad()

        metrics = {}

        bce = nn.BCEWithLogitsLoss()

        #####################
        # Train the generator
        #####################

        x_batch, y_batch = batch

        x_batch = x_batch.to(self.rank)
        y_batch = y_batch.to(self.rank)

        ones = torch.ones((x_batch.size(0), 1)).float().to(self.rank)

        g_loss_dict = OrderedDict({})

        z_batch = self.gen.sample_z(x_batch.size(0)).to(self.rank)
        # r_x.detach() because the generator (decoder)
        # training is separate from SimSiam.
        x_fake = self.gen(z_batch, y_batch)

        # x_fake = G(r1, z)
        # we want D(x_fake|r{1,2}) to be 'real'
        d_out, d_out_predz, d_out_predy = self.disc(x_fake, y_batch)

        g_d_loss = bce(d_out, ones)
        d_mi_y_loss = torch.mean((d_out_predy - y_batch) ** 2)
        d_mi_z_loss = torch.mean((d_out_predz - z_batch) ** 2)

        g_loss_dict["g_d_loss"] = (1.0, g_d_loss)
        g_loss_dict["g_mi_y_loss"] = (self.gamma / 2., d_mi_y_loss)
        g_loss_dict["g_mi_z_loss"] = (self.gamma / 2., d_mi_z_loss)

        if train:
            g_total_loss, g_total_loss_str = self._eval_loss_dict(g_loss_dict)

            if self.iteration == 0:
                logger.info(
                    "{}: G is optimising this total loss: {}".format(
                        self.rank, g_total_loss_str
                    )
                )
                logger.debug("x_batch.shape = {}".format(x_batch.shape))
                logger.debug(
                    "x_batch min max = {}, {}".format(x_batch.min(), x_batch.max())
                )
                logger.debug("x_fake.shape = {}".format(x_fake.shape))
                logger.debug(
                    "x_fake min max = {}, {}".format(x_fake.min(), x_fake.max())
                )

            if self.iteration % self.update_g_every == 0:
                g_total_loss.backward()
                self.opt_g.step()
                self.update_ema()

        #if self.iteration % 100 == 0:
        #    print(g_loss_dict)

        #########################
        # Train the discriminator
        #########################

        self.opt_d.zero_grad()

        d_loss_dict = OrderedDict({})

        bce = nn.BCEWithLogitsLoss()
        zeros = torch.zeros((x_batch.size(0), 1)).float().to(self.rank)
        ones = torch.ones((x_batch.size(0), 1)).float().to(self.rank)

        d_out_real, _, _ = self.disc(x_batch, 
                                     y_batch)
        d_out_fake, d_out_predz, d_out_predy = self.disc(x_fake.detach(), 
                                                         y_batch)

        # mi_loss_rz = torch.mean((d_out_fake_rz[1] - x_fake_z) ** 2)
        d_loss_real = bce(d_out_real, ones)
        d_loss_fake = bce(d_out_fake, zeros)
        d_loss = (d_loss_real + d_loss_fake) / 2.0

        d_mi_y_loss = torch.mean((d_out_predy - y_batch) ** 2)
        d_mi_z_loss = torch.mean((d_out_predz - z_batch) ** 2)

        d_loss_dict["d_loss"] = (1.0, d_loss)
        d_loss_dict["d_mi_y_loss"] = (self.gamma / 2., d_mi_y_loss)
        d_loss_dict["d_mi_z_loss"] = (self.gamma / 2., d_mi_z_loss)

        if train:
            d_total_loss, d_total_loss_str = self._eval_loss_dict(d_loss_dict)
            if self.iteration == 0:
                logger.info(
                    "{}: D is optimising this total loss: {}".format(
                        self.rank, d_total_loss_str
                    )
                )

            d_total_loss.backward()
            self.opt_d.step()

        self.iteration += 1

        # TODO: make this morboe efficient
        with torch.no_grad():
            metrics = {k: v[1].detach() for k, v in g_loss_dict.items()}
            metrics.update({k: v[1].detach() for k, v in d_loss_dict.items()})

            #metrics.update({k: v.detach() for k, v in g_metrics.items()})
            # metrics['r_loss_neg'] = r_loss_neg.detach()
            # metrics['r_acc'] = r_acc.detach()

        return metrics

    def vis_on_loader(
        self, loader, savedir, split, n_batches=1, aux_loader=None, **kwargs
    ):
        for i, batch in enumerate(loader):
            self.vis_on_batch(
                batch, savedir=savedir, split=split, dataset=loader.dataset
            )
            break

    def get_state_dict(self):
        state_dict = {
            "gen": self.gen.state_dict(),
            "disc": self.disc.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }
        if self.use_ema:
            state_dict["gen_ema"] = self.gen_ema.state_dict()

        return state_dict

    def _load_state_dict_with_mismatch(self, current_model_dict, chkpt_model_dict):
        # https://github.com/pytorch/pytorch/issues/40859
        # strict won't let you load in a state dict with
        # mismatch param shapes, so we do this hack here.
        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
            for k, v in zip(current_model_dict.keys(), chkpt_model_dict.values())
        }
        return new_state_dict

    def set_state_dict(self, state_dict, load_opt=True, strict=True):

        if strict:
            self.gen.load_state_dict(state_dict["gen"], strict=strict)
            if self.use_ema:
                self.gen_ema.load_state_dict(state_dict["gen_ema"], strict=strict)
        else:
            self.gen.load_state_dict(
                self._load_state_dict_with_mismatch(
                    current_model_dict=self.gen.state_dict(),
                    chkpt_model_dict=state_dict["gen"],
                )
            )
            if self.use_ema:
                self.gen_ema.load_state_dict(
                    self._load_state_dict_with_mismatch(
                        current_model_dict=self.gen_ema.state_dict(),
                        chkpt_model_dict=state_dict["gen_ema"],
                    )
                )
        self.disc.load_state_dict(state_dict["disc"], strict=strict)
        if load_opt:
            self.opt_g.load_state_dict(state_dict["opt_g"])
            self.opt_d.load_state_dict(state_dict["opt_d"])
        # self.opt_p.load_state_dict(state_dict['opt_p'])
