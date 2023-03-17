import torch
from torch import nn
from torch import optim
from torch.distributions import Normal

from typing import List, Union, Dict

import numpy as np

from ..utils import suppress_stdout_stderr, compute_realism_score

from ..setup_logger import get_logger
logger = get_logger(__name__)

from itertools import chain

from tqdm import tqdm

class BackproppableLatents(nn.Module):
    def __init__(self, 
                 dataset, 
                 model, 
                 classifier,
                 fid_stats,
                 N: int = 100,
                 test_dataset = None,
                 gamma: float = 0.0,
                 beta: float = 0.0, 
                 prior_mu: float = 0.0, 
                 prior_std: float = 1.0,
                 lr: float = 2e-3):
        super().__init__()
        self.dataset = dataset
        self.test_dataset = test_dataset

        # Initialise ys as a tensor. Just select 100 random
        # y's from the validation dataset.
        ys = torch.clone(dataset.y)
        randperm = torch.randperm(ys.size(0))[0:N]
        self.ys = nn.Parameter(ys[randperm], requires_grad=True)
        zs = model.sample_z(self.ys.size(0)).normal_(prior_mu, prior_std)
        self.Z = nn.Parameter(zs, requires_grad=True)

        self.N = N
        self.model = model
        self.classifier = classifier
        self.fid_stats = fid_stats
        self.opt = optim.Adam( (self.Z, self.ys), lr=lr)
        self.beta = beta

        self.gamma = gamma
        self.prior_mu = nn.Parameter(torch.zeros(self.Z.size(1))+prior_mu)
        self.prior_std = nn.Parameter(torch.zeros(self.Z.size(1))+prior_std)
        self.prior = Normal(self.prior_mu, self.prior_std)
        self.lr = lr


        #import pdb; pdb.set_trace()

        logger.info("prior = {}".format(self.prior))

    def _eval_loss_dict(self, loss_dict):
        # TODO: make this class part of base model
        loss = 0.0
        loss_str = []
        for key, val in loss_dict.items():
            if len(val) != 2:
                raise Exception("val must be a tuple of (coef, loss)")
            if val[0] != 0:
                # Only add the loss if the coef is != 0
                loss += val[0] * val[1]
            loss_str.append("{} * {}".format(val[0], key))
        return loss, (" + ".join(loss_str))

    def fit(
        self,
        n_iters: int = 10000,
        eval_every: int = 500,
    ) -> List[Dict]:
        """
        Args:
            n_iters: number of iterations to train for.
            eval_every: evaluate metrics this many iterations.

        Returns:
            a list of dictionaries, where each dictionary comes from the model's
              score_on_dataset() method (evaluated every `eval_every` iters).
        """

        losses = []
        metrics = []
        for b in tqdm(range(n_iters), desc=str(self.__class__)):

            loss_dict = {}

            # y's are normalised by default in the dataset, so
            # generate a denormed version.
            ys_denorm = self.dataset.denorm_y(self.ys)

            # Sample a random batch
            self.opt.zero_grad()
            x_sampled = self.model.gen(self.Z, self.ys)
        
            pred_y = self.classifier.predict(x_sampled) # <-- not denormed
            test_pred_y = self.dataset.predict(x_sampled) # <-- is denormed
            agreement = torch.mean( (pred_y - self.ys)**2 )
            pz = -self.prior.log_prob(self.Z)

            # want to do:
            # argmin_{z,y} \ -y + beta*agreement(z,y,valid_oracle)
            loss_dict['ys'] = (self.beta, -self.ys.mean())
            loss_dict['agr'] = (1.0, agreement.mean())
            loss_dict['prior'] = (self.gamma, pz.mean())

            this_loss, this_loss_str = self._eval_loss_dict(loss_dict)
            this_loss.backward()

            with torch.no_grad():
                pred_y_denorm = self.dataset.denorm_y(pred_y)
                agg_denorm = torch.mean( (pred_y_denorm-ys_denorm)**2 )

            #test_pred_y_denorm = self.dataset.denorm_y(test_pred_y)

            with torch.no_grad():
                test_agg_denorm = np.mean( (test_pred_y-ys_denorm.cpu().numpy())**2 )


            if eval_every is not None and b % eval_every == 0:
                logger.info(this_loss_str)
                metrics.append({
                    'ys': self.dataset.denorm_y(self.ys).mean().item(),
                    
                    'pred_ys': pred_y_denorm.mean().item(), 
                    'agr': agg_denorm.item(),
                    
                    'test_pred_ys': test_pred_y.mean().item(),
                    'test_agr': test_agg_denorm,
                })
                logger.info(metrics[-1])
    
            self.opt.step()

            """
            if eval_every is not None and b % eval_every == 0:
                metrics.append(self.model.score_on_dataset(
                    self.dataset,
                    self.classifier,
                    self.fid_stats['valid'],
                    # Ignore y and just return x_sampled
                    sample_fn=lambda y: x_sampled,
                    prefix=self.dataset.split,
                    verbose=False
                ))
                if self.test_dataset is not None:
                    metrics.append(self.model.score_on_dataset(
                        self.test_dataset,
                        self.classifier,
                        self.fid_stats['test'],
                        # Ignore y and just return x_sampled
                        sample_fn=lambda y: x_sampled,
                        prefix=self.test_dataset.split,
                        verbose=False
                    ))
                
                logger.info(metrics[-1]["valid_agg_denorm"])
                logger.info(metrics[-1]["valid_density"])                
                logger.info(metrics[-1]["valid_precision"])                
            """

        return metrics