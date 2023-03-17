import os
from typing import Callable, Dict
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
# from torch.cuda.amp import autocast, GradScaler
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

from torch.optim import AdamW as Adam

from torch.utils.data import DataLoader, Dataset
from ..classifier import Classifier

from collections import OrderedDict

from ... import utils as ut

from ...setup_logger import get_logger
logger = get_logger(__name__)

from ..base_model import BaseModel
from .networks import EncoderDecoder
#from .nerf_helpers import 

from torch.distributions import Normal, kl_divergence

class VAE(BaseModel):

    DEFAULT_ARGS = {}

    def _validate_args(self, dd):
        pass

    def __init__(
        self,
        n_out: int,
        gen_kwargs: Dict,
        optim_kwargs: Dict,
        rank: int = 0,
        beta: float = 0.0,
        use_ema: bool = False,
        ema_rate: float = 0.9999,
        verbose: bool = True,
    ):
        super().__init__()

        self.rank = rank

        self.gen = EncoderDecoder(n_out=n_out, **gen_kwargs)
        self.gen.to(self.rank)

        self._ema_rate = ema_rate

        if use_ema:
            self.gen_ema = EncoderDecoder(n_out=n_out, **gen_kwargs)
            self.gen_ema.to(self.rank)
        else:
            self.gen_ema = None

        if verbose and self.rank == 0:
            logger.info("gen: {}".format(self.gen))
            logger.info("# gen params: {}".format(ut.count_params(self.gen)))

        beta1 = optim_kwargs.pop("beta1")
        beta2 = optim_kwargs.pop("beta2")
        self.opt_g = Adam(self.gen.parameters(), betas=(beta1, beta2), **optim_kwargs)

        logger.info(self.opt_g)

        self.use_ema = use_ema
        self.beta = beta

    @property
    def ema_src_network(self):
        return self.gen

    @property
    def ema_tgt_network(self):
        return self.gen_ema

    @property
    def ema_rate(self):
        return self._ema_rate

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

    def eval(self):
        self.gen.eval()

    def _run_on_batch(self, batch, train=True, verbose=False, **kwargs):

        if train:
            self.train()
        else:
            self.eval()

        if train:
            self.opt_g.zero_grad()

        metrics = {}
        g_loss_dict = OrderedDict({})

        x_batch, y_batch = batch
        x_batch = x_batch.to(self.rank)
        y_batch = y_batch.to(self.rank)

        qz_x = self.gen.encode(x_batch, y_batch)
        this_z = qz_x.rsample()
        
        pz = Normal(torch.zeros_like(this_z), torch.ones_like(this_z))            
        kl_loss = kl_divergence(qz_x, pz).mean()

        p_x_given_zy = self.gen.decode(this_z, y_batch)
        sampled_x = p_x_given_zy.rsample()

        # HACK: I have no idea how log_prob works for RelaxedOneHot,
        # so just hack the solution here with nn.CrossEntropy()
        if type(p_x_given_zy) == RelaxedOneHotCategorical:
            cce_loss = nn.NLLLoss() # takes log probs and class integers

            # e.g. (bs,8,4) -> (bs,8) -> (bs*8,)
            this_labels = x_batch.reshape(-1, *self.gen.discrete_dims).\
                argmax(-1).flatten()
            # sampled_x is (bs,8,4) -> (bs*8, 4)
            nll_loss = cce_loss(
                torch.log(sampled_x).view(-1, sampled_x.size(-1)),
                this_labels
            )
        else:
            nll_loss = -p_x_given_zy.log_prob(x_batch).mean()

        with torch.no_grad():
            # if we use rsample() here then it's gonna be highly
            # stochastic because sd=1, so take the mean from
            # p(x|y,z)
            if hasattr(p_x_given_zy, 'loc'):
                recon_loss = torch.mean((p_x_given_zy.loc-x_batch)**2)
                metrics["recon_loss"] = recon_loss
            
            elbo = nll_loss + kl_loss
            metrics["elbo"] = elbo

            perm = torch.randperm(x_batch.size(0))
            p_x_given_z1y = self.gen.decode(this_z[perm], y_batch)
            p_x_given_zy1 = self.gen.decode(this_z, y_batch[perm])

            if hasattr(p_x_given_zy, 'loc'):
                # Difference when decoding with a different z
                metrics["z_diff"] = torch.mean(
                    (p_x_given_z1y.loc-p_x_given_zy.loc)**2
                )
                # Difference when decoding with a different y
                metrics["y_diff"] = torch.mean(
                    (p_x_given_zy1.loc-p_x_given_zy.loc)**2
                )

        g_loss_dict["nll_loss"] = (1.0, nll_loss)
        g_loss_dict["kl_loss"] = (self.beta, kl_loss)
        g_total_loss, g_total_loss_str = self._eval_loss_dict(g_loss_dict)
        if verbose:
            logger.info(
                "{}: G is optimising this total loss: {}".format(
                    self.rank, g_total_loss_str
                )
            )
            logger.debug("x_batch.shape = {}".format(x_batch.shape))
            logger.debug(
                "x_batch min max = {}, {}".format(x_batch.min(), x_batch.max())
            )
            logger.debug("x_fake.shape = {}".format(sampled_x.shape))
            logger.debug(
                "x_fake min max = {}, {}".format(sampled_x.min(), sampled_x.max())
            )


        if train:
            g_total_loss.backward()
            self.opt_g.step()
            self.update_ema()

        if verbose:
            logger.info("g_loss_dict: {}".format(g_loss_dict))
            logger.info("g_metrics: {}".format(metrics))

        with torch.no_grad():
            metrics = {k: v.detach() for k, v in metrics.items()}
            metrics.update({k: v[1].detach() for k, v in g_loss_dict.items()})

        return metrics

    def score_on_dataset(self, 
                         dataset: Dataset, 
                         classifier: Classifier, 
                         fid_stats: Dict,
                         sample_fn: Callable = None,
                         prefix: str = "", 
                         **kwargs):
        """Evaluate all metrics on dataset.

        Args:
            dataset:
            classifier: features will be computed from this
            fid_stats: fid stats
            prefix: metrics will be prefixed with this
        """

        score_dict = super().score_on_dataset(
            dataset=dataset,
            classifier=classifier,
            sample_fn=sample_fn,
            fid_stats=fid_stats,
            prefix=prefix,
            **kwargs
        )

        full_loader = DataLoader(dataset, batch_size=1)
        eval_stats = self.eval_on_loader(full_loader, return_buf=True)

        elbo = eval_stats['elbo']
        logger.debug("""
        elbo for bs=1:
        min: {}, max: {}, sd = {}
        """.format(elbo.min(), elbo.max(), elbo.std()))
        
        score_dict["{}_elbo_score".format(prefix)] = eval_stats["elbo"].mean()
        
        return score_dict

    def get_state_dict(self):
        state_dict = {
            "gen": self.gen.state_dict(),
            "opt_g": self.opt_g.state_dict(),
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
        if load_opt:
            self.opt_g.load_state_dict(state_dict["opt_g"])