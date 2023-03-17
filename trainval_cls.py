import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import time
import json

from haven import haven_wizard as hw

#from src.fid import fid_score
#from src.models import GAN
from src.models import Classifier # NoiseClassifier
from src.datasets import get_dataset

#from src.datasets import DuplicateDatasetMTimes
from src.utils import (validate_and_insert_defaults, 
                       Argument,
                       DuplicateDatasetMTimes,
                       load_json_from_file,
                       get_checkpoint,
                       discretise_into_bins)

from src import setup_logger
logger = setup_logger.get_logger(__name__)

NONETYPE = type(None)
DEFAULTS = {
    "dataset": Argument("dataset", "TFBind8", [str]),
    "dataset_M": Argument("dataset_M", 0, [int]),

    "use_noise": Argument("use_noise", False, [bool]),

    # If true, train on the entire dataset (still set aside X% for
    # internal validation). Set this when there is already a ground
    # truth oracle that comes with the dataset.
    "test_oracle": Argument("test_oracle", False, [bool]),

    "gain": Argument("gain", 1.0, [float]),
    "gain_y": Argument("gain_y", None, [float, NONETYPE]),

    "postprocess": Argument("postprocess", False, [bool]),
    "oracle": Argument("oracle", "ResNet-v0", [str]),
    "batch_size": Argument("batch_size", 512, [int]),
    "epochs": Argument("epochs", 5000, [int]),
    "model_kwargs": Argument("model_kwargs", {}, [dict]),

    # This is meant to be used if one chooses the ClassifierGuidance
    # class, which requires some extra args like the number of timesteps
    # and what noise schedule to use.
    "classifier_kwargs": Argument("classifier_kwargs", {}, [dict]),
   
    "optim_kwargs": {
        "lr": Argument("lr", 2e-4, [float]),
        "beta1": Argument("beta1", 0.0, [float]),
        "beta2": Argument("beta2", 0.9, [float]),
        "weight_decay": Argument("weight_decay", 0.0, [float])
    },
    "save_every": Argument("save_every", None, [int, NONETYPE]),
    "eval_every": Argument("eval_every", 10, [int])
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


def validate_args(dd):
    pass

def trainval(exp_dict, savedir, args):

    validate_and_insert_defaults(exp_dict, DEFAULTS)
    validate_args(exp_dict)

    logger.info("ENV flags: {}".format(os.environ))

    # When haven saves exp_dict.json, it does not consider keys (default keys)
    # inserted after the experiment launches. So save a new exp dict
    # that overrides the one produced by Haven.
    with open("{}/exp_dict.json".format(savedir), "w") as f:
        f.write(json.dumps(exp_dict))

    test_oracle = exp_dict['test_oracle']

    # Load dataset
    train_dataset = get_dataset(task_name=exp_dict['dataset'],
                                oracle_name=exp_dict['oracle'],
                                split='train', 
                                gain=exp_dict["gain"],
                                gain_y=exp_dict["gain_y"])
    valid_dataset = get_dataset(task_name=exp_dict['dataset'],
                                oracle_name=exp_dict['oracle'],
                                split='valid',
                                gain=exp_dict["gain"],
                                gain_y=exp_dict["gain_y"])
    if not test_oracle:
        # By default, the train and validation sets are merged together
        # and a small internal validation set is set aside.
        dataset = ConcatDataset((train_dataset, valid_dataset))
        rnd_state = np.random.RandomState(0)
        indices = np.arange(0, len(dataset))
        rnd_state.shuffle(indices)
        train_indices = indices[0:int(0.95*len(indices))]
        valid_indices = indices[int(0.95*len(indices))::]

        train_dataset = Subset(dataset, indices=train_indices)
        valid_dataset = Subset(dataset, indices=valid_indices)
    else:
        # This should be set if there is no exact GT oracle. When there is no exact
        # GT oracle, we should train one using the _full_ dataset.
        logger.warning("Training using the full dataset, so training test oracle...")
        test_dataset = get_dataset(task_name=exp_dict['dataset'],
                                   oracle_name=exp_dict['oracle'],
                                   split='test',
                                   gain=exp_dict["gain"])
        dataset = ConcatDataset((train_dataset, valid_dataset, test_dataset))
        train_dataset = dataset
        valid_dataset = dataset

    dataset_M = exp_dict["dataset_M"]
    if dataset_M > 0:
        logger.info("Duplicating dataset M={} times".format(dataset_M))
        train_dataset_ = DuplicateDatasetMTimes(train_dataset, M=dataset_M)
    else:
        train_dataset_ = train_dataset

    logger.debug("len of train: {}".format(len(train_dataset)))
    logger.debug("len of valid: {}".format(len(valid_dataset)))

    train_loader = DataLoader(train_dataset_,
                              shuffle=True,
                              batch_size=exp_dict['batch_size'])

    valid_loader = DataLoader(valid_dataset,
                              shuffle=True,
                              batch_size=exp_dict['batch_size'])

    if exp_dict["use_noise"]:
        #model = NoiseClassifier(
        #    model_kwargs=exp_dict['model_kwargs'],
        #    optim_kwargs=exp_dict['optim_kwargs'],
        #    **exp_dict["classifier_kwargs"]
        #)
        raise NotImplementedError()
    else:
        model = Classifier(
            model_kwargs=exp_dict['model_kwargs'],
            optim_kwargs=exp_dict['optim_kwargs'],
            **exp_dict["classifier_kwargs"]
        )

    best_metric = np.inf
    chk_metric = "valid_loss"
    max_epochs = exp_dict["epochs"]
    eval_every = exp_dict["eval_every"]
    score_list = []
    for epoch in range(0, max_epochs):

        t0 = time.time()

        score_dict = {}
        score_dict["epoch"] = epoch

        #if train_sampler is not None:
        #    train_sampler.set_epoch(epoch)
        #if dev_sampler is not None:
        #    dev_sampler.set_epoch(epoch)

        # (1) Train GAN.
        train_dict_ = model.train_on_loader(
            train_loader,
            #epoch=epoch,
            #pbar=world_size <= 1,
            #pbar=False if 'DISABLE_PBAR' in os.environ else True
        )
        train_dict = {("train_" + key): val for key, val in train_dict_.items()}

        valid_dict_ = model.val_on_loader(
            valid_loader,
            #epoch=epoch,
            #pbar=world_size <= 1,
            #pbar=False if 'DISABLE_PBAR' in os.environ else True
        )
        valid_dict = {("valid_" + key): val for key, val in valid_dict_.items()}      

        score_dict.update(train_dict)
        score_dict.update(valid_dict)
        score_dict["time"] = time.time() - t0

        if eval_every > 0 and epoch % eval_every == 0 and epoch > 0:

            logger.info(train_dict)
            logger.info(valid_dict)

            if score_dict[chk_metric] < best_metric:
                logger.info(
                    "new best metric: from {}={:.4f} to {}={:.4f}".format(
                        chk_metric, best_metric, chk_metric, score_dict[chk_metric]
                    )
                )
                best_metric = score_dict[chk_metric]
                hw.save_checkpoint(
                    savedir,
                    fname_suffix="." + chk_metric,
                    score_list=score_list,
                    model_state_dict=model.get_state_dict(),
                    verbose=False,
                )

        score_list += [score_dict]

    print("Experiment completed")