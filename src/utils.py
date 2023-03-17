import numpy as np
from haven import haven_utils as hu
import numpy as np
import os
import json
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import Dataset
from typing import Dict, List
from prdc import prdc

from .setup_logger import get_logger

logger = get_logger(__name__)


def load_json_from_file(filename):
    return json.loads(open(filename, "r").read())


class Argument:
    def __init__(self, name, default, types, choices=None):
        """_summary_

        Args:
            name (_type_): name of the argument
            default (_type_): its default value
            types (_type_): a list of allowed types
            choices: the only allowable values that can be taken
        """
        self.name = name
        self.default = default
        self.types = types
        self.choices = choices

    def validate(self, x):
        if type(x) not in self.types:
            raise Exception(
                "argument {} has invalid type: {}, allowed types = {}".format(
                    self.name, type(x), self.types
                )
            )
        if self.choices is not None:
            if x not in self.choices:
                raise Exception(
                    "argument {} has value {} but allowed values = {}".format(
                        self.name, x, self.choices
                    )
                )


class DuplicateDatasetMTimes(Dataset):
    """Cleanly duplicate a dataset M times. This is to avoid
    the massive overhead associated with data loader resetting
    for small dataset sizes, e.g. the support set which only
    has k examples per class.
    """

    def __init__(self, dataset, M):
        self.dataset = dataset
        self.N_actual = len(dataset)
        self.M = M

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx % self.N_actual)

    def __len__(self):
        return self.N_actual * self.M


def str2bool(st):
    if st.lower() == "true":
        return True
    return False


def get_checkpoint(savedir, return_model_state_dict=False, map_location=None):
    chk_dict = {}

    # score list
    score_list_fname = os.path.join(savedir, "score_list.pkl")
    if os.path.exists(score_list_fname):
        score_list = hu.load_pkl(score_list_fname)
    else:
        score_list = []

    chk_dict["score_list"] = score_list
    if len(score_list) == 0:
        chk_dict["epoch"] = 0
    else:
        chk_dict["epoch"] = score_list[-1]["epoch"] + 1

    model_state_dict_fname = os.path.join(savedir, "model.pth")
    if return_model_state_dict:
        if os.path.exists(model_state_dict_fname):
            chk_dict["model_state_dict"] = hu.torch_load(
                model_state_dict_fname, map_location=map_location
            )

        else:
            chk_dict["model_state_dict"] = {}

    return chk_dict


def validate_and_insert_defaults(exp_dict, defaults, ignore_recursive=[]):
    """Inserts default values into the exp_dict.

    Will raise an exception if exp_dict contains
    a key that is not recognised. If the v for a
    (k,v) pair is also a dict, this method will
    recursively call insert_defaults() into that
    dictionary as well.

    Args:
        exp_dict (dict): dictionary to be added to
        ignore_recursive: any key in here will not be validated
          recursively.
    """
    ignore_recursive = set(ignore_recursive)

    # First make a pass through exp_dict and see if there are any
    # unknown keys. This involves recursing into nested dictionaries
    # if necessary.
    for key in exp_dict.keys():
        if key not in defaults:
            # Check if there are any unknown keys.
            print(exp_dict)
            raise Exception(
                "Found key in exp_dict but is not recognised: {}".format(key)
            )
        else:
            if type(defaults[key]) == dict and key not in ignore_recursive:
                # If this key maps to a dict, then apply
                # this function recursively
                logger.debug(
                    ">> validate_and_insert_defaults: recurse into {}".format(key)
                )
                validate_and_insert_defaults(exp_dict[key], defaults[key])

    # Make a pass through exp_dict, and insert any default values as
    # needed. This involves recursing into nested dictionaries
    # if necesary.
    _recursive_insert_defaults(exp_dict, defaults)


def _recursive_insert_defaults(exp_dict, defaults):
    for k, v in defaults.items():
        # If the key is not in exp_dict, then we will insert it
        # by default, unless v is itself a dictionary, then we
        # have to recurse into it.
        if k not in exp_dict:
            if type(v) is dict:
                exp_dict[k] = {}
                logger.debug(">> _recurse_insert_defaults: recurse into {}".format(k))
                _recursive_insert_defaults(exp_dict[k], v)
            else:
                logger.info(
                    "Inserting default arg into exp dict: {}={}".format(k, v.default)
                )
                # print(" exp dict is:", exp_dict)
                exp_dict[k] = v.default

                # sanity check, the default value should match the allowable types
                defaults[k].validate(v.default)
        else:
            # is this block necessary?
            # validate the arg
            if type(v) is not dict:
                # print(k, v)
                defaults[k].validate(exp_dict[k])


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def count_params(module, trainable_only=True):
    """Count the number of parameters in a
    module.
    :param module: PyTorch module
    :param trainable_only: only count trainable
      parameters.
    :returns: number of parameters
    :rtype:
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num


def discretise_into_bins(arr, n_bins):
    bins = np.array(
        KBinsDiscretizer(strategy="quantile", n_bins=n_bins)
        .fit(arr)
        .transform(arr)
        .todense()
        .argmax(1)
        .flatten()
    )[0]
    return bins


# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def dict2name(x: Dict, keys: List[str], sep: str = ",") -> str:
    """Convert a dictionary to a string representation based on a list of provided keys"""
    return sep.join([str(a) + "=" + str(b) for a, b in x.items() if a in keys])


def compute_realism_score(real_features, fake_features, nearest_k):
    """Realism score, from Section 5 of Improved PR metrics paper"""
    real_nearest_neighbour_distances = prdc.compute_nearest_neighbour_distances(
        real_features, nearest_k=nearest_k
    )
    # Do not take the max as done in the paper, they do something
    # that is more akin to a median actually.
    result = np.median(
        np.expand_dims(real_nearest_neighbour_distances, axis=0)
        / prdc.compute_pairwise_distance(fake_features, real_features)
    , axis=1) #.max(axis=1)
    return result