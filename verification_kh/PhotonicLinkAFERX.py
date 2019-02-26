from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Sequence, Optional

if TYPE_CHECKING:
    from bag.core import Testbench

from bag.simulation.core import TestbenchManager, MeasurementManager
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from verification_kh.GenericACMM import GenericACMM
from verification_kh.TIA import TIATBM
import itertools
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import scipy.integrate as integ
import pdb


class PhotonicLinkAFERXMM(MeasurementManager):

    def __init__(self, *args, **kwargs):
        MeasurementManager.__init__(self, *args, **kwargs)
        self.overall_results = {}

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'tia_diff'

    def get_testbench_info(self,  # type: MeasurementManager
                           state,  # type: str
                           prev_output,  # type: Optional[Dict[str, Any]]
                           ):
        tb_type = state
        tb_name = self.get_testbench_name(tb_type)
        tb_specs = self.get_testbench_specs(tb_type).copy()
        tb_specs['sim_vars']['ibias_in'] = self.specs['ibias_in']
        tb_params = self.get_default_tb_sch_params(tb_type)

        return tb_name, tb_type, tb_specs, tb_params

    def process_output(self, state, data, tb_manager):
        # type: (str, Dict[str, Any], TIAMM) -> Tuple[bool, str, Dict[str, Any]]

        done = True
        next_state = ''

        if state == 'tia_diff':
            self.post_process_diff(state, data, tb_manager)
            next_state = 'tia_cm'
            done = False

        elif state == 'tia_cm':
            self.post_process_cm(state, data, tb_manager)
            next_state = ''
            done = True

        return done, next_state, self.overall_results

    def post_process_diff(self, state, data, tb_manager):
        ax = plt.gca()
        self.add_plot(state, data, 'outpTIA', ax=ax, title='R_tia', function=lambda x: 20*np.log10(2*np.abs(x)), save=False)
        self.add_plot(state, data, 'outdiff', ax=ax, title='R_tia_ctle', function=lambda x: 20 * np.log10(np.abs(x)),
                      save=True)
        self.add_plot(state, data, 'input_noise', title='input_noise', function=lambda x: np.abs(x))
        self.add_plot(state, data, 'out_tran', x_axis='time', log_axis='none', title='out_tran',
                      function=lambda x: x)

        ac_res = tb_manager.get_R_and_f3db(data, ['outdiff'])
        # tran_res = tb_manager.get_tset(data, out_name='out_tran', input_name='in_tran',
        #                                tot_err=self.specs['tset_tol'], gain=ac_res['R_TIA_outdiff'], plot_flag=False)
        eye_height = self.get_eye_height_approximation(data, ['out_tran'], tstop=tb_manager.specs['sim_vars']['tstop'],
                                                       Tbit=self.specs['Tbit'])
        f3db_list = ac_res['f3db_outdiff']
        noise_res = tb_manager.get_integrated_noise(data, ['input_noise'], f3db_list)

        output = dict(
            r_afe=ac_res['R_TIA_outdiff'],
            f3db=f3db_list,
            rms_input_noise=noise_res['rms_input_noise'],
            ibias=np.abs(data['ibias']),
            # tset=tran_res['tset_out_tran'],
            eye_height=eye_height,
        )

        self.overall_results.update(**output)

    def get_eye_height_approximation(self, data, out_names, tstop, Tbit, thresh=2e-5):
        # ignoring outer sweeps like corners
        for out_name in out_names:
            out = np.abs(data[out_name])
            time_max_out = data['time'][np.argmax(out)]
            time_max_out = min(time_max_out, Tbit)

            f_out = interp.interp1d(data['time'], out, kind='cubic')

            eye_height = f_out(time_max_out)
            v = eye_height
            sample_time = time_max_out + Tbit
            while sample_time < tstop:
                v = f_out(sample_time)
                eye_height -= abs(v)
                sample_time += Tbit

            return eye_height


    def post_process_cm(self, state, data, tb_manager):
        self.add_plot(state, data, 'outcm', title='cm_cm', function=lambda x: 20 * np.log10(np.abs(x)))
        self.add_plot(state, data, 'outdiff', title='cm_diff', function=lambda x: 20 * np.log10(np.abs(x)))
        ac_res = tb_manager.get_gain(data, ['outcm', 'outdiff'])
        output = dict(
            cmcm_gain=ac_res['gain_outcm'],
            cmdm_gain=ac_res['gain_outdiff'],
            cmrr=self.overall_results['r_afe']/ac_res['gain_outcm'],
        )
        self.overall_results.update(**output)

    def add_plot(self, state, data, y_axis, function=lambda x: x, x_axis='freq', ax=None, title=None, log_axis='x', save=True, show=False):
        """
        this function should plot the data and maybe save it if needed. It depends on the MeasurementManager subclass.
        For more comlpex ploting function it should be overwritten and use state variable for conditioning
        :param state:
        :param data:
        :param y_axis:
        :param x_axis:
        :param ax:
        :param title:
        :param save:
        :param show:
        :param log_axis: 'x'|'y'|'both'|'none'
        :return:
        """

        functions_dict = {'x': plt.semilogx, 'y': plt.semilogx, 'both': plt.loglog, 'none': plt.plot}
        #TODO: Unfinished

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        if title is None:
            title = y_axis

        # import IPython
        sweep_kwrds = data['sweep_params'][y_axis]
        sweep_kwrds = [kwrd for kwrd in sweep_kwrds if kwrd != x_axis]
        # combos = itertools.product(*(list(range(len(data[swp_kwrd]))) for swp_kwrd in sweep_kwrds))

        # IPython.embed()
        # if combos:
        #     for index in combos:
        #         functions_dict[log_axis](data[x_axis], np.abs(data[y_axis][index, :]))
        # else:
        functions_dict[log_axis](data[x_axis], function(data[y_axis]))
        plt.ylabel(y_axis)
        plt.xlabel(x_axis)
        ax.grid()
        if save:
            fname = os.path.join(self.data_dir, title + ".png")
            if os.path.isfile(fname):
                os.remove(fname)
            plt.savefig(fname, dpi=200)
            plt.close()
        if show:
            plt.show()






