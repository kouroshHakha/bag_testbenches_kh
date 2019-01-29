
"""
This module defines an generic Measurment manager for any amplifier architecture, with voltage input and
voltage output. For more complicated mangement of testbench results one should use subclasses of this module
in which they override the basic functionalities.
"""

from bag.simulation.core import MeasurementManager, TestbenchManager
import numpy as np
import IPython
import pdb
import matplotlib.pyplot as plt
import itertools
import os
from bag.io.sim_data import save_sim_results

class GenericACMM(MeasurementManager):

    def __init__(self, *args, **kwargs):
        MeasurementManager.__init__(self, *args, **kwargs)
        self.overall_results = {}

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'ac_forward'

    def process_output(self, state, data, tb_manager):
        # type: (str, Dict[str, Any], TestbenchManager) -> Tuple[bool, str, Dict[str, Any]]
        """Process simulation output data.

        Parameters
        ----------
        state : str
            the current FSM state
        data : Dict[str, Any]
            simulation data dictionary.
        tb_manager : GenericACTB
            the testbench manager object.

        Returns
        -------
        done : bool
            True if this measurement is finished.
        next_state : str
            the next FSM state.
        output : Dict[str, Any]
            a dictionary containing post-processed data.
        """
        if state == 'ac_forward':
            self.add_plot(state, data, 'outdiff', title='ac_forward')
            self.run_ac_forward_post_process(data, tb_manager)

        elif state == 'ac_common_mode':
            self.add_plot(state, data, 'outcm', title=state)
            self.run_common_mode_post_process(data, tb_manager)

        elif state == 'ac_power_supply':
            self.add_plot(state, data, 'outcm', title=state)
            self.run_power_supply_post_process(data, tb_manager)

        next_state = self.get_next_state(state)
        done = self.is_done(next_state)

        return done, next_state, self.overall_results

    def is_done(self, state):
        return state == ''

    def get_next_state(self, state):
        if state == 'ac_forward':
            next_state = 'ac_common_mode' if 'ac_common_mode' in self.specs['testbenches'] else ''
        elif state == 'ac_common_mode':
            next_state = 'ac_power_supply' if 'ac_power_supply' in self.specs['testbenches'] else ''
        elif state == 'ac_power_supply':
            next_state = ''
        else:
            raise ValueError('Unknown state: %s' % state)
        return next_state

    def run_ac_forward_post_process(self, data, tb_manager):

        output_dict = tb_manager.get_gain_and_w3db(data, ['outdiff'])
        results = dict(
            dc_gain=output_dict['gain_outdiff'],
            f3db=1/2/np.pi*output_dict['w3db_outdiff'],
            ibias=data['ibias'],
            corners=data['corner'],
        )
        self.overall_results.update(**results)

        return results

    def run_common_mode_post_process(self, data, tb_manager):

        output_dict = tb_manager.get_gain_and_w3db(data, ['outcm'])
        # preprocess cm gain before computing cmrr, clip it to some epsilon if it's too small
        output_dict['gain_outcm'] = np.clip(output_dict['gain_outcm'], a_min=1.0e-10, a_max=None)
        # compute cmrr
        dc_gain = self.overall_results['dc_gain']
        cmrr_db = 20 * np.log10(dc_gain / output_dict['gain_outcm'])
        results = dict(
            gain_cm=output_dict['gain_outcm'],
            cmrr_db=cmrr_db,
            corners=data['corner'],
        )
        self.overall_results.update(results)

    def run_power_supply_post_process(self, data, tb_manager):

        output_dict = tb_manager.get_gain_and_w3db(data, ['outcm'])
        # preprocess power supply gain before computing psrr, clip it to some epsilon if it's too small
        output_dict['gain_outcm'] = np.clip(output_dict['gain_outcm'], a_min=1.0e-10, a_max=None)
        # compute psrr
        dc_gain = self.overall_results['dc_gain']
        psrr_db = 20 * np.log10(dc_gain / output_dict['gain_outcm'])
        results = dict(
            gain_ps=output_dict['gain_outcm'],
            psrr_db=psrr_db,
            corners=data['corner'],
        )
        self.overall_results.update(**results)

    def add_plot(self, state, data, y_axis, x_axis='freq', ax=None, title=None, save=True, show=False):
        """
        this function should plot the data and maybe save it if needed. It depends on the MeasurementManager subclass.
        :param state:
        :param data:
        :param y_axis:
        :param x_axis:
        :param ax:
        :param title:
        :param save:
        :param show:
        :return:
        """

        #TODO: Unfinished

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        if title is None:
            title = y_axis

        sweep_kwrds = data['sweep_params'][y_axis]
        sweep_kwrds = [kwrd for kwrd in sweep_kwrds if kwrd != x_axis]
        combos = itertools.product(*(data[swp_kwrd] for swp_kwrd in sweep_kwrds))

        # for values in zip(combos):
        plt.grid()
        plt.plot(np.log(data[x_axis]), np.abs(data[y_axis][0]), label='ff')
        plt.plot(np.log(data[x_axis]), np.abs(data[y_axis][1]), label='tt')
        plt.ylabel(y_axis)
        plt.xlabel(x_axis)
        if save:
            fname = os.path.join(self.data_dir, title + ".png")
            if os.path.isfile(fname):
                os.remove(fname)
            plt.savefig(fname, dpi=200)
        if show:
            plt.show()
