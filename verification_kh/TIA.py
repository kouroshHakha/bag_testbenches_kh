from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Sequence, Optional

if TYPE_CHECKING:
    from bag.core import Testbench

from bag.simulation.core import TestbenchManager, MeasurementManager
import matplotlib.pyplot as plt
import os
from scipy import interpolate
import numpy as np

class TIATBM(TestbenchManager):
    def __init__(self,
                 data_fname,  # type: str
                 tb_name,  # type: str
                 impl_lib,  # type: str
                 specs,  # type: Dict[str, Any]
                 sim_view_list,  # type: Sequence[Tuple[str, str]]
                 env_list,  # type: Sequence[str]
                 ):
        # type: (...) -> None
        TestbenchManager.__init__(self, data_fname, tb_name, impl_lib, specs,
                                  sim_view_list, env_list)
    def setup_testbench(self, tb):
        # not done properly, safer to make them equal
        sim_vars = self.specs.get('sim_vars', None)
        sim_outputs = self.specs.get('sim_outputs', None)

        print(sim_vars)
        for key, value in sim_vars.items():
            tb.set_parameter(key, value)
        if sim_outputs is not None:
            for key, val in sim_outputs.items():
                tb.add_output(key, val)

    @classmethod
    def add_plot(self, data, yaxis_key=None, xaxis_key='time'):
        if yaxis_key is None:
            raise ValueError('yaxis_key should be specified')
        if yaxis_key not in data:
            raise ValueError('yaxis_key = {} not found in data keywords'.format(yaxis_key))
        plt.plot(data[xaxis_key], data[yaxis_key])

    @classmethod
    def save_plot(self, fname):
        plt.grid()
        if os.path.isfile(fname):
            os.remove(fname)
        plt.savefig(fname, dpi=200)
        plt.close()


class TIAMM(MeasurementManager):

    def __init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list, env_list):
        # type: (str, str, str, Dict[str, Any], Dict[str, str], Sequence[Tuple[str, str]], Sequence[str]) -> None
        MeasurementManager.__init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list, env_list)

        testbenches_tia = specs['testbenches']['tia']

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'tia'

    def process_output(self, state, data, tb_manager):
        # type: (str, Dict[str, Any], TIAMM) -> Tuple[bool, str, Dict[str, Any]]
        done = True
        next_state = ''

        plt.semilogx(data['freq'], np.abs(data['input_noise']))
        plt.title("Input Noise")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude (A/sqrt(Hz))")

        fname = os.path.join(self.data_dir, 'in_noise.png')
        if os.path.isfile(fname):
            os.remove(fname)
        plt.savefig(fname)
        plt.gcf().clear()
        plt.close()

        plt.semilogx(data['freq'], np.abs(data['output_noise']))
        plt.title("Output Noise")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude (V/sqrt(Hz))")

        fname = os.path.join(self.data_dir, 'out_noise.png')
        if os.path.isfile(fname):
            os.remove(fname)
        plt.savefig(fname)
        plt.gcf().clear()
        plt.close()

        #tb_manager.add_plot(data, yaxis_key='iin', xaxis_key='freq')
        #tb_manager.save_plot(os.path.join(self.data_dir, 'plot2.png'))

        output = dict()

        return done, next_state, output





