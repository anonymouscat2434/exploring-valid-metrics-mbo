import os
import torch
from torch import nn
from torch.utils.data import DataLoader

# Parallel stuff
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import time
import json

# from collections import Counter

from haven import haven_utils as hu
from haven import haven_wizard as hw

# from src.fid import fid_score
from src.models import GAN, VAE, Diffusion, Classifier, Bprop
from src.fid import fid_score
from src.datasets import get_dataset

# from src.datasets import DuplicateDatasetMTimes
from src.utils import (
    validate_and_insert_defaults,
    load_json_from_file,
    get_checkpoint,
    DuplicateDatasetMTimes,
    Argument
)

from src import setup_logger
logger = setup_logger.get_logger(__name__)

import tensorflow
print("TENSORFLOW VERSION", tensorflow.__version__)


NONETYPE = type(None)
DEFAULTS = {
    "model": Argument("model", "gan", [str], ["gan", "vae", "diffusion", "bprop"]),
    "dataset": Argument("dataset", "TFBind8", [str]),
    "oracle": Argument("oracle", "ResNet-v0", [str]),
    "dataset_M": Argument("dataset_M", 0, [int]),
    "gain": Argument("gain", 1.0, [float]),
    "gain_y": Argument("gain_y", None, [float, NONETYPE]),
    
    "postprocess": Argument("postprocess", False, [bool]),
    "pretrained_oracle": Argument("pretrained_oracle", None, [str, NONETYPE]),
    "batch_size": Argument("batch_size", 512, [int]),
    "N_linspace": Argument("N_linspace", 100, [int]),

    # These kwargs get passed into classifier.features(), which is the
    # method that extracts features from a particular dataset.
    "fid_kwargs": {
        "all_features": Argument("all_features", True, [bool])
    },

    # Args that are specific to diffusion.
    "diffusion_kwargs": Argument("diffusion_kwargs", None, [dict, NONETYPE]),
    
    #"fid_N_bins": Argument("fid_N_bins", 10, [int]),
    
    "epochs": Argument("epochs", 5000, [int]),
    "gen_kwargs": Argument("gen_kwargs", None, [dict, NONETYPE]),
    "disc_kwargs": Argument("disc_kwargs", None, [dict, NONETYPE]),
    # This is just a convenience arg for use with Haven,
    # it means we can just do gen_kwargs := eval(gen_kwargs_str).
    # This will override the actual gen_kwargs dict if this is
    # set.
    "gen_kwargs_str": Argument("gen_kwargs_str", None, [str, NONETYPE]),
    "optim_kwargs": {
        "lr": Argument("lr", 2e-4, [float]),
        "betas": Argument("betas", (0.0, 0.9), [tuple, list]),
        "weight_decay": Argument("weight_decay", 0.0, [float])
    },
    "save_every": Argument("save_every", None, [int, NONETYPE]),
    "eval_every": Argument("eval_every", 10, [int]),
    "eval_batch_size": Argument("eval_batch_size", None, [int, NONETYPE]),

    # If set to True, eval ground truth oracle in score_on_dataset.
    # One may want to disable this in the event that evaluating the
    # ground truth oracle during train/val takes too much time.
    "eval_gt": Argument("eval_gt", True, [bool]),

    # This is a very important flag. DC seems to give really good values
    # only a few epochs into training, which is not good. We want to only
    # initiate early stopping checkpoints after `eval_after` epochs have
    # passed.
    "eval_after": Argument("eval_after", 10, [int]),

    # Log metrics every this many seconds. Default is 30 sec.
    "log_every": Argument("log_every", 30, [int, NONETYPE]),
    
    "update_g_every": Argument("update_g_every", None, [NONETYPE, int]),
    "gamma": Argument("gamma", 1.0, [float]),
    "beta": Argument("beta", 1.0, [float]),
    "use_ema": Argument("use_ema", True, [bool]),
    "ema_rate": Argument("ema_rate", 0.9999, [float]),
    "valid_metrics": Argument(
        "valid_metrics", ["valid_fid_ema", "valid_agg_denorm_ema"], [list, tuple]
    ),
    # dummy variable, does not serve any purpose
    "id": Argument("id", None, [int, NONETYPE]),
}

class FidWrapper(nn.Module):
    """Just a wrapper that conforms to the same
    interface as the Inception model used to
    compute FID.
    """

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return [self.f(x).unsqueeze(-1).unsqueeze(-1)]


def extra_validate_args(dd):
    # Load model and get checkpoint
    if dd["gen_kwargs_str"] is not None:
        logger.info("gen_kwargs_str is set, overriding gen_kwargs...")
        dd["gen_kwargs"] = eval(dd["gen_kwargs_str"])
    del dd["gen_kwargs_str"]
    if dd["model"] == "vae":
        if dd["disc_kwargs"] is not None:
            logger.warning("disc_kwargs is only defined for when model==gan")
        if dd["update_g_every"] is not None:
            logger.warning("update_g_every is only defined for when model==gan")
        if dd["gamma"] is not None:
            logger.warning("gamma is only defined for when model==gan")
    if dd["eval_batch_size"] is None:
        dd["eval_batch_size"] = dd["batch_size"]

def trainval(exp_dict, savedir, args):
    validate_and_insert_defaults(exp_dict, DEFAULTS)
    extra_validate_args(exp_dict)

    # When haven saves exp_dict.json, it does not consider keys (default keys)
    # inserted after the experiment launches. So save a new exp dict
    # that overrides the one produced by Haven.
    with open("{}/exp_dict.json".format(savedir), "w") as f:
        f.write(json.dumps(exp_dict))

    #world_size = int(os.environ["WORLD_SIZE"])
    #if world_size == 0:
    #logger.info("WORLD_SIZE==0, running on a single process")
    _trainval(rank=0, exp_dict=exp_dict, savedir=savedir)
    #else:
    #    logger.info("WORLD_SIZE>0, running on multiprocess...")
    #    mp.spawn(
    #        _trainval,
    #        args=(world_size, exp_dict, savedir, args),
    #        nprocs=world_size,
    #        join=True,
    #    )


@torch.no_grad()
def compute_fid_reference_stats(dataset, classifier, return_features=False, **fid_kwargs):
    tgt_X_cuda = dataset.X.to(classifier.rank)
    tgt_X_features = classifier.features(tgt_X_cuda, **fid_kwargs).cpu().numpy()
    logger.debug("tgt_X_features: {}".format(tgt_X_features.shape))
    tgt_mu, tgt_sigma = fid_score.calculate_activation_statistics(tgt_X_features)
    if return_features:
        return tgt_X_features, (tgt_mu, tgt_sigma)
    return tgt_mu, tgt_sigma


def _init_from_args(rank, exp_dict, skip_model=False):
    """Return train/valid/test datasets and the model.
    This is a convenience class so that it can be called from
    other methods like eval.py.

    Args:
      rank: gpu rank
      exp_dict: the experiment dictionary
      skip_model: if true, do not return the model. Simply return
        Bprop, which is the 'dumb' model.
    """
    # Load dataset
    train_dataset = get_dataset(
        task_name=exp_dict["dataset"],
        oracle_name=exp_dict["oracle"],
        split="train",
        gain=exp_dict["gain"],
        gain_y=exp_dict["gain_y"]
    )

    valid_dataset = get_dataset(
        task_name=exp_dict["dataset"],
        oracle_name=exp_dict["oracle"],
        split="valid",
        gain=exp_dict["gain"],
        gain_y=exp_dict["gain_y"]

    )
    test_dataset = get_dataset(
        task_name=exp_dict["dataset"],
        oracle_name=exp_dict["oracle"],
        split="test",
        gain=exp_dict["gain"],
        gain_y=exp_dict["gain_y"]

    )

    name2model = {
        "vae": VAE,
        "gan": GAN,
        "diffusion": Diffusion,
    }

    
    model_class = name2model[exp_dict["model"]]
    base_kwargs = dict(
        n_out=train_dataset.n_in,
        optim_kwargs=exp_dict["optim_kwargs"],
        use_ema=exp_dict["use_ema"],
        ema_rate=exp_dict["ema_rate"],
        rank=rank,
    )

    if skip_model:
        model = Bprop(
            classifier = None,
            **base_kwargs
        )
    else:   
        if model_class == GAN:
            model = model_class(
                gen_kwargs=exp_dict["gen_kwargs"],
                disc_kwargs=exp_dict["disc_kwargs"],            # TODO gen_kwargs
                update_g_every=exp_dict["update_g_every"],      # TODO gan_kwargs
                gamma=exp_dict["gamma"],                        # TODO gan_kwargs
                **base_kwargs
            )
        elif model_class == VAE:
            model = model_class(
                gen_kwargs=exp_dict["gen_kwargs"],
                beta=exp_dict["beta"],          # TODO this needs to be a vae_kwarg
                **base_kwargs
            )
        elif model_class == Diffusion:
            diffusion_kwargs = exp_dict["diffusion_kwargs"]
            model = model_class(
                n_classes=diffusion_kwargs["n_classes"],
                tau=diffusion_kwargs["tau"],
                w=diffusion_kwargs["w"],
                gen_kwargs=exp_dict["gen_kwargs"],
                **base_kwargs
            )
        else:
            raise Exception()

    return (train_dataset, valid_dataset, test_dataset), model

def _trainval(rank, exp_dict, savedir):

    datasets, model = _init_from_args(rank, exp_dict)
    train_dataset, valid_dataset, test_dataset = datasets

    if type(model) is Bprop:
        raise ValueError("Bprop can only be used at inference time")

    # Load our own pre-trained oracle for FID
    # computation purposes.
    cls_exp_dict = load_json_from_file(
        "{}/exp_dict.json".format(os.path.dirname(exp_dict["pretrained_oracle"]))
    )
    cls = Classifier(
        model_kwargs=cls_exp_dict["model_kwargs"],
        optim_kwargs=cls_exp_dict["optim_kwargs"],
    )
    cls.set_state_dict(torch.load(exp_dict["pretrained_oracle"]))

    # Explicitly set what gpu to put the weights on.
    # If map_location is not set, each rank (gpu) will
    # load these onto presumably gpu0, causing an OOM
    # if we run this code under a resuming script.
    chk_dict = get_checkpoint(
        savedir,
        return_model_state_dict=True,
        map_location=lambda storage, loc: storage.cuda(rank),
    )
    if len(chk_dict["model_state_dict"]):
        model.set_state_dict(chk_dict["model_state_dict"], strict=True)

    # Safety here since older experiments didn't have these
    # args set.
    cls_gain = cls_exp_dict.get("gain", 1.0)
    cls_gain_y = cls_exp_dict.get("gain_y", None)

    if cls_exp_dict["postprocess"] != exp_dict["postprocess"]:
        raise ValueError(
            "Postprocess flags are inconsistent. This would cause bad "
            + "FID estimates since real / fake x statistics would be inconsistent"
        )
    elif cls_gain != exp_dict["gain"]:
        raise ValueError("Classifier experiment has a different gain (cls={} vs {})".\
            format(cls_gain, exp_dict["gain"]))
    elif cls_gain_y != exp_dict["gain_y"]:
        raise ValueError("Classifier experiment has a different gain_y (cls={} vs {})".\
            format(cls_gain_y, exp_dict["gain_y"]))

    dataset_M = exp_dict["dataset_M"]
    if dataset_M > 0:
        train_dataset_ = DuplicateDatasetMTimes(train_dataset, M=dataset_M)
    else:
        train_dataset_ = train_dataset
    train_loader = DataLoader(
        train_dataset_, shuffle=True, batch_size=exp_dict["batch_size"],
        #pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, shuffle=True, batch_size=exp_dict["batch_size"],
        #pin_memory=True
    )

    # ---------------------------------------------
    # Pre-compute FID stats for valid and test sets
    # ---------------------------------------------

    fid_stats = {}
    for tgt_dataset, tgt_name in zip(
        [train_dataset, valid_dataset, test_dataset], ["train", "valid", "test"]
    ):
        logger.info("Computing FID reference stats for: {}".format(tgt_name))
        fid_stats[tgt_name] = compute_fid_reference_stats(
            tgt_dataset, cls, True, **exp_dict["fid_kwargs"]
        )

    # ------------------
    # Run Train-Val loop
    # ------------------
    
    max_epochs = exp_dict["epochs"]
    save_every = exp_dict["save_every"]
    eval_every = exp_dict["eval_every"]
    eval_after = exp_dict["eval_after"]

    valid_metrics = exp_dict["valid_metrics"]
    record_metrics = {k: np.inf for k in valid_metrics}
    logger.info("metrics for checkpoint saving: {}".format(valid_metrics))
    if len(chk_dict["score_list"]) == 0:
        for key in record_metrics.keys():
            record_metrics[key] = np.inf
    else:
        # If we're resuming from a pre-trained checkpoint, find what the
        # minimum value is meant to be for each of the metrics in
        # `valid_metrics`.
        for key in record_metrics.keys():
            this_scores = [
                score[key] for score in chk_dict["score_list"] if key in score
            ]
            if len(this_scores) == 0:
                record_metrics[key] = np.inf
            else:
                record_metrics[key] = min(this_scores)
                logger.debug("record_metrics[{}] = {}".format(key, min(this_scores)))


    logger.info("Starting epoch: {}".format(chk_dict["epoch"]))
    for epoch in range(chk_dict["epoch"], max_epochs):

        t0 = time.time()

        #score_dict.update(
        """
        model.score_on_dataset(
            dataset=train_dataset, 
            classifier=cls, 
            fid_stats=fid_stats["train"], 
            fid_kwargs=exp_dict["fid_kwargs"],
            eval_gt=exp_dict["eval_gt"],
            prefix="train",
            batch_size=exp_dict["eval_batch_size"]
        )
        """
        #)

        # TODO: reduce cpu stats as well???
        if rank == 0:
            score_dict = {}
            score_dict["epoch"] = epoch

        # if train_sampler is not None:
        #    train_sampler.set_epoch(epoch)
        # if dev_sampler is not None:
        #    dev_sampler.set_epoch(epoch)

        # (1) Train GAN.
        train_dict_ = model.train_on_loader(
            train_loader,
            epoch=epoch,
            savedir=savedir,
            log_every=exp_dict["log_every"],
            # pbar=world_size <= 1,
            pbar=False if "DISABLE_PBAR" in os.environ else True,
        )
        train_dict = {("train_" + key): val for key, val in train_dict_.items()}

        valid_dict_ = model.eval_on_loader(
            valid_loader,
            epoch=epoch,
            savedir=savedir,
            log_every=exp_dict["log_every"],
            # pbar=world_size <= 1,
            pbar=False if "DISABLE_PBAR" in os.environ else True,
        )
        valid_dict = {("valid_" + key): val for key, val in valid_dict_.items()}

        score_dict.update(train_dict)
        score_dict.update(valid_dict)
        score_dict["time"] = time.time() - t0

        if eval_every > 0 and epoch % eval_every == 0 and epoch > eval_after:

            for this_dataset, this_prefix in [
                (train_dataset, "train"), 
                (valid_dataset, "valid"), 
                (test_dataset, "test")
            ]:
            
                score_dict.update(
                    model.score_on_dataset(
                        dataset=this_dataset, 
                        classifier=cls, 
                        fid_stats=fid_stats[this_prefix], 
                        fid_kwargs=exp_dict["fid_kwargs"],
                        eval_gt=exp_dict["eval_gt"],
                        prefix=this_prefix,
                        batch_size=exp_dict["eval_batch_size"]
                    )
                )

            chk_dict["score_list"] += [score_dict]

            for metric in record_metrics.keys():
                if score_dict[metric] < record_metrics[metric]:
                    logger.info(
                        "New best metric {}: from {:.4f} to {:.4f}".format(
                            metric, record_metrics[metric], score_dict[metric]
                        )
                    )
                    # save the new best metric
                    record_metrics[metric] = score_dict[metric]
                    hw.save_checkpoint(
                        savedir,
                        fname_suffix="." + metric,
                        score_list=chk_dict["score_list"],
                        model_state_dict=model.get_state_dict(),
                        verbose=False,
                    )
        else:
            chk_dict["score_list"] += [score_dict]

        # Save checkpoint
        hw.save_checkpoint(
            savedir,
            score_list=chk_dict["score_list"],
            model_state_dict=model.get_state_dict(),
            verbose=False,
        )

        # If `save_every` is defined, save every
        # this many epochs.
        if save_every is not None:
            if epoch > 0 and epoch % save_every == 0:
                hw.save_checkpoint(
                    savedir,
                    fname_suffix="." + str(epoch),
                    score_list=chk_dict["score_list"],
                    model_state_dict=model.get_state_dict(),
                    verbose=False,
                )

    print("Experiment completed")
