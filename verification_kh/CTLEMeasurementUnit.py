from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Sequence, Optional

from bag.simulation.core import MeasurementManager, TestbenchManager
import numpy as np
from scipy import interpolate
import IPython
import pdb
import matplotlib.pyplot as plt
import itertools
import os
from bag.io.sim_data import save_sim_results

if TYPE_CHECKING:
    from verification_ec.ac.core import ACTB

from verification_kh.GenericACMM import GenericACMM

class CTLEMeasurementManager(GenericACMM):

    def run_ac_forward_post_process(self, data, tb_manager):
        output_dict = tb_manager.get_dc_gain_max_gain_first_pole(data, ['outdiff'])
        results = dict(
            dc_gain=output_dict['dc_gain_outdiff'],
            max_gain=output_dict['max_gain_outdiff'],
            first_pole=output_dict['first_pole_outdiff'],
            ibias=data['ibias'],
            corners=data['corner'],
        )

        self.overall_results.update(**results)