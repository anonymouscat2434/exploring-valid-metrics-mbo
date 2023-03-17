import subprocess
import os
from haven import haven_utils as hu
from haven import haven_wizard as hw

from src import setup_logger
logger = setup_logger.get_logger(__name__)

def trainval(exp_dict, savedir, args):
    cmd_to_run = exp_dict['cmd']
    logger.debug("Run command: {}".format(cmd_to_run))
    subprocess.run(cmd_to_run, shell=True)