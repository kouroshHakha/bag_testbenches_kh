from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Sequence, Optional

if TYPE_CHECKING:
    from bag.core import Testbench

from bag.simulation.core import TestbenchManager, MeasurementManager
import matplotlib.pyplot as plt
import os
from scipy import interpolate
import numpy as np

class ODTBM(TestbenchManager):
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
        if sim_vars is not None:
            # just forcing tr be equal to tf in case it's not
            sim_vars['tr'] = sim_vars['tf']
            sim_vars['td'] = sim_vars['Tper']/4
        sim_outputs = self.specs.get('sim_outputs', None)

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


class ODMM(MeasurementManager):

    def __init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list, env_list):
        # type: (str, str, str, Dict[str, Any], Dict[str, str], Sequence[Tuple[str, str]], Sequence[str]) -> None
        MeasurementManager.__init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list, env_list)

        testbenches_od = specs['testbenches']['od']
        self.c_wait = testbenches_od['sim_vars']['c_wait']
        self.Tper = testbenches_od['sim_vars']['Tper']
        self.tsetup = self.specs['tsetup']

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'od'

    def process_output(self, state, data, tb_manager):
        # type: (str, Dict[str, Any], ODMM) -> Tuple[bool, str, Dict[str, Any]]
        # TODO: make this work for multiple corners
        done = True
        next_state = ''

        '''
        # Sanity check with visualization
        tb_manager.add_plot(data, yaxis_key='inclk')
        tb_manager.add_plot(data, yaxis_key='outdiff')
        tb_manager.add_plot(data, yaxis_key='indiff')
        tb_manager.save_plot(os.path.join(self.data_dir, 'plot.png'))
        '''

        # fit vout = f(time)
        # read value of vout @ different times
        vout = data['outdiff']
        time = data['time']

        fvout = interpolate.interp1d(time, vout, kind='cubic')
        t_charge = self.c_wait * self.Tper - self.tsetup
        t_reset = (self.c_wait+0.5) * self.Tper - self.tsetup
        t_out = (self.c_wait + 1) * self.Tper - self.tsetup

        v_charge = fvout(t_charge)
        v_reset = fvout(t_reset)
        v_out = fvout(t_out)
        ibias = np.abs(data['ibias'])

        tb_manager.add_plot(data, yaxis_key='inclk')
        tb_manager.add_plot(data, yaxis_key='outdiff')
        tb_manager.add_plot(data, yaxis_key='indiff')
        tb_manager.save_plot(os.path.join(self.data_dir, 'plot.png'))

        output = dict(v_charge=np.float(v_charge),
                      v_reset=np.float(v_reset),
                      v_out=np.float(v_out),
                      ibias=np.float(ibias))

        return done, next_state, output





