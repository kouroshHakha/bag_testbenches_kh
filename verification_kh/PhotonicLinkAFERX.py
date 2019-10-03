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
        # ax = plt.gca()
        # self.add_plot(state, data, 'outpTIA', ax=ax, title='R_tia', function=lambda x: 20*np.log10(2*np.abs(x)), save=False)
        # self.add_plot(state, data, 'outdiff', ax=ax, title='R_tia_ctle', function=lambda x: 20 * np.log10(np.abs(x)),
        #               save=True)
        # self.add_plot(state, data, 'input_noise', title='input_noise', function=lambda x: np.abs(x))
        # self.add_plot(state, data, 'out_tran', x_axis='time', log_axis='none', title='out_tran',
        #               function=lambda x: x)

        ac_res = tb_manager.get_R_and_f3db_new(data, 'outdiff', fig_loc=self.data_dir)
        # if you want to run settling time simulation (i.e. inject a small signal step and measure the settling time to
        # the total desired percentage error value uncomment the tb_manager.get_test command, comment the get_eye_height
        # command and also in the yaml file make sure the pulse width is 500ms. Also don't

        tran_res = None
        # tran_res = tb_manager.get_tset_new(data, output_name='out_tran', input_name='in_tran',
        #                                    tot_err=self.specs['tset_tol'], gain_list=ac_res['gain'], plot_loc='./tset_debug.png')

        eye_char = None
        # if you want the eye_height make sure the settling time is commented and pulse width in the yaml file is Tbit.
        eye_char = self.get_eye_height_approximation(data, 'out_tran', tstop=tb_manager.specs['sim_vars']['tstop'],
                                                     Tbit=self.specs['Tbit'], tmargin=self.specs['tmargin'], fig_loc=self.data_dir)
        f3db_list = ac_res['f3db']
        input_noise, output_noise = tb_manager.get_integrated_noise_new(data, ['input_noise', 'output_noise'], f3db_list, fig_loc=self.data_dir)

        if data['ibias'].shape:
            ibias=  np.abs(data['ibias']).tolist()
        else:
            ibias = [np.abs(data['ibias']).tolist()]

        output = dict(
            corner=ac_res['corner'],
            r_afe=ac_res['gain'],
            f3db=f3db_list,
            rms_input_noise=input_noise['integ_noise'],
            rms_output_noise=output_noise['integ_noise'],
            ibias=ibias,
        )

        if tran_res:
            output['tset'] = tran_res['tset']

        if eye_char:
            eye_level_thickness_ratio = []
            for height in eye_char['height']:
                if height == 0:
                    eye_level_thickness_ratio.append(1)
                else:
                    index = eye_char['height'].index(height)
                    eye_level_thickness_ratio.append(eye_char['level_thickness'][index]/eye_char['height'][index])

            output.update(dict(eye_height=eye_char['height'],
                               eye_span_height=eye_char['span_height'],
                               eye_level_thickness = eye_char['level_thickness'],
                               eye_level_thickness_ratio = eye_level_thickness_ratio,
                               ))

        self.overall_results.update(**output)

    def get_eye_height_approximation(self, data, output_name, tstop, Tbit, tmargin, thresh=2e-5, fig_loc=None):
        axis_names = ['corner', 'time']
        height_list, span_height_list, level_thickness_list = list(), list(), list()

        sweep_vars = data['sweep_params'][output_name]
        swp_corner = ('corner' in sweep_vars)
        output_corner = data[output_name]
        if not swp_corner:
            corner_list = [self.env_list[0]]
            sweep_vars = ['corner'] + sweep_vars
            output_corner = output_corner[None, :]
        else:
            corner_list = data['corner'].tolist()

        order = [sweep_vars.index(swp) for swp in axis_names]
        output_corner = np.transpose(output_corner, axes=order)

        for corner, v_output  in zip(corner_list, output_corner):
            if fig_loc:
                plt.plot(data['time'], v_output)
                plt.savefig(os.path.join(fig_loc, 'tran_out_{}'.format(corner)), dpi=200)
                plt.close()

            out = np.abs(v_output)
            time_max_out = data['time'][np.argmax(out)]
            time_max_out = min(time_max_out, Tbit)

            f_out = interp.interp1d(data['time'], out, kind='cubic')

            # compute the main eye height
            sample_time = time_max_out + Tbit
            eye_level_thickness=0
            while sample_time < tstop:
                v = f_out(sample_time)
                eye_level_thickness+=abs(v)
                sample_time += Tbit

            eye_height = f_out(time_max_out) - eye_level_thickness
            if eye_height < 0 :
                eye_height = 0

            # compute the left marginal eye height
            left_eye_height = f_out(time_max_out-tmargin)
            sample_time = time_max_out - tmargin + Tbit
            while sample_time < tstop:
                v = f_out(sample_time)
                left_eye_height-=abs(v)
                sample_time += Tbit

            # compute the right marginal eye height
            right_eye_height = f_out(time_max_out+tmargin)
            sample_time = time_max_out + tmargin + Tbit
            while sample_time < tstop:
                v = f_out(sample_time)
                right_eye_height-=abs(v)
                sample_time += Tbit

            # the overall eye_span_height is going to be the minimum of the right and left one
            eye_span_height = min(left_eye_height, right_eye_height)
            if eye_span_height < 0:
                eye_span_height = 0

            height_list.append(float(eye_height))
            span_height_list.append(float(eye_span_height))
            level_thickness_list.append(float(eye_level_thickness))

        eye_char = dict(
            corner=corner_list,
            height=height_list,
            span_height=span_height_list,
            level_thickness=level_thickness_list,
        )

        return eye_char


    def post_process_cm(self, state, data, tb_manager):
        # self.add_plot(state, data, 'outcm', title='cm_cm', function=lambda x: 20 * np.log10(np.abs(x)))
        # self.add_plot(state, data, 'outdiff', title='cm_diff', function=lambda x: 20 * np.log10(np.abs(x)))
        # outcm_res, outdiff_res = tb_manager.get_R_and_f3db_new(data, ['outcm', 'outdiff'], fig_loc=self.data_dir)
        outcm_res, outdiff_res = tb_manager.get_gain_new(data, ['outcm', 'outdiff'], fig_loc=self.data_dir)
        output = dict(
            cmcm_gain=outcm_res['gain'],
            cmdm_gain=outdiff_res['gain'],
            cmrr=(np.array(self.overall_results['r_afe'])/np.array(outcm_res['gain'])).tolist(),
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






