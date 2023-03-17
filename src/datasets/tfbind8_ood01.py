from .tfbind8 import TfBind8

import numpy as np

from ..setup_logger import get_logger
logger = get_logger(__name__)

class TfBind8_OOD01(TfBind8):

    def mask_dataset(self, x: np.ndarray, y: np.ndarray):
        """Generate a boolean mask denoting instances that should be marked
        as invalid (out-of-distribution).

        Args:
            x (np.ndarray):
            y (np.ndarray):

        Returns:
            _type_: _description_
        """
        # For each instance, if the number of 0's is > 3 then say f(x)==0
        #x_as_strs = [ "".join(s.astype('str')) for s in x ]
        mask = ( (x==0).sum(1) > 3 ) | ( (x==1).sum(1) > 3 )
        
        logger.debug("% of instances marked as invalid: {} %".format(mask.mean()*100.))
        return mask