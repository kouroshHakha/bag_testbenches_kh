from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Sequence, Optional

if TYPE_CHECKING:
    from bag.core import Testbench

from bag.simulation.core import TestbenchManager, MeasurementManager
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from verification_kh.GenericACMM import GenericACMM
import itertools
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import scipy.integrate as integ
import pdb


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

        # print(sim_vars)
        if sim_vars is not None:
            for key, value in sim_vars.items():
                tb.set_parameter(key, value)
        if sim_outputs is not None:
            if sim_outputs is not None:
                for key, val in sim_outputs.items():
                    tb.add_output(key, val)

    @classmethod
    def get_R_and_f3db(cls, data, output_list, output_dict=None):
        # type: (Dict[str, Any], List[str], Optional[Dict[str, Any]]) -> Dict[str, Any]
        """Returns a dictionary of gain and 3db bandwidth information.

        Parameters
        ----------
        data : Dict[str, Any]
            the simulation data dictionary.
        output_list : Sequence[str]
            list of output names to compute gain/bandwidth for.
        output_dict : Optional[Dict[str, Any]]
            If not None, append to the given output dictionary instead.

        Returns
        -------
        output_dict : Dict[str, Any]
            A BAG data dictionary containing the gain/bandwidth information.
        """
        if output_dict is None:
            output_dict = {}
        swp_info = data['sweep_params']
        f_vec = data['freq']
        for out_name in output_list:
            out_arr = data[out_name]
            swp_params = swp_info[out_name]
            freq_idx = swp_params.index('freq')
            new_swp_params = [par for par in swp_params if par != 'freq']
            gain_arr, f3db_arr = cls._compute_R_and_f3db(f_vec, np.abs(out_arr), freq_idx)
            cls.record_array(output_dict, data, gain_arr, 'R_TIA_' + out_name, new_swp_params)
            cls.record_array(output_dict, data, f3db_arr, 'f3db_' + out_name, new_swp_params)

        return output_dict

    @classmethod
    def get_gain(cls, data, output_list, output_dict=None):
        # type: (Dict[str, Any], List[str], Optional[Dict[str, Any]]) -> Dict[str, Any]
        """Returns a dictionary of gain bandwidth information.

        Parameters
        ----------
        data : Dict[str, Any]
            the simulation data dictionary.
        output_list : Sequence[str]
            list of output names to compute gain/bandwidth for.
        output_dict : Optional[Dict[str, Any]]
            If not None, append to the given output dictionary instead.

        Returns
        -------
        output_dict : Dict[str, Any]
            A BAG data dictionary containing the gain/bandwidth information.
        """
        if output_dict is None:
            output_dict = {}
        swp_info = data['sweep_params']
        f_vec = data['freq']
        for out_name in output_list:
            out_arr = data[out_name]
            swp_params = swp_info[out_name]
            freq_idx = swp_params.index('freq')
            new_swp_params = [par for par in swp_params if par != 'freq']
            out_arr = np.abs(out_arr)
            out_arr = np.moveaxis(out_arr, freq_idx, -1)
            gain_arr = out_arr[..., 0]
            cls.record_array(output_dict, data, gain_arr, 'gain_' + out_name, new_swp_params)

        return output_dict

    @classmethod
    def get_integrated_noise(cls, data, output_list, bandwidth_arr, output_dict=None):
        # type: (Dict[str, Any], List[str], np.ndarray, Optional[Dict[str, Any]]) -> Dict[str, Any]
        """Returns a dictionary of gain and 3db bandwidth information.

        Parameters
        ----------
        data : Dict[str, Any]
            the simulation data dictionary.
        output_list : Sequence[str]
            list of output names to compute gain/bandwidth for.
        bandwidth_arr: np.ndarray
            list of bandwidths for every swept parameter (in units of Hz)
        output_dict : Optional[Dict[str, Any]]
            If not None, append to the given output dictionary instead.

        Returns
        -------
        output_dict : Dict[str, Any]
            A BAG data dictionary containing the gain/bandwidth information.
        """

        if output_dict is None:
            output_dict = {}
        swp_info = data['sweep_params']
        f_vec = data['freq']
        for out_name in output_list:
            out_density = data[out_name]
            swp_params = swp_info[out_name]
            freq_idx = swp_params.index('freq')
            new_swp_params = [par for par in swp_params if par != 'freq']

            bandwidth_arr = np.expand_dims(bandwidth_arr, axis=0)
            rms_arr = cls._compute_integrated_noise(f_vec, np.abs(out_density), bandwidth_arr, freq_idx)
            cls.record_array(output_dict, data, rms_arr, 'rms_' + out_name, new_swp_params)

        return output_dict

    @classmethod
    def get_tset(cls, data, out_name, input_name, tot_err, gain, output_dict=None, plot_flag=False):

        if output_dict is None:
            output_dict = {}
        swp_info = data['sweep_params']
        time = data['time']

        swp_params = swp_info[out_name]
        time_idx = swp_params.index('time')
        new_swp_params = [par for par in swp_params if par != 'time']
        tset_arr = cls._compute_tset(time, data[out_name], data[input_name], tot_err, gain, time_idx, plot_flag)
        cls.record_array(output_dict, data, tset_arr, 'tset_' + out_name, new_swp_params)

        return output_dict


    @classmethod
    def _compute_R_and_f3db(cls, f_vec, out_arr, freq_idx):
        # type: (np.ndarray, np.ndarray, int) -> Tuple[np.ndarray, np.ndarray]
        """Compute the DC R_TIA and bandwidth of the amplifier given output array.

        Parmeters
        ---------
        f_vec : np.ndarray
            the frequency vector.  Must be sorted.
        out_arr : np.ndarray
            the amplifier output transfer function.  Could be multidimensional.
        freq_idx : int
            frequency axis index.

        Returns
        -------
        gain_arr : np.ndarray
            the DC gain array.
        f3db_arr : np.ndarray
            the 3db bandwidth array.  Contains NAN if the transfer function never
            intersect the gain.
        """
        # move frequency axis to last axis
        out_arr = np.moveaxis(out_arr, freq_idx, -1)
        gain_arr = out_arr[..., 0]

        # convert
        orig_shape = out_arr.shape
        num_pts = orig_shape[-1]
        out_log = 20 * np.log10(out_arr.reshape(-1, num_pts))
        gain_log_3db = 20 * np.log10(gain_arr.reshape(-1)) - 3

        # find first index at which gain goes below gain_log 3db
        diff_arr = out_log - gain_log_3db[:, np.newaxis]
        idx_arr = np.argmax(diff_arr < 0, axis=1)
        freq_log = np.log10(f_vec)
        freq_log_max = freq_log[idx_arr]

        num_swp = out_log.shape[0]
        f3db_list = []
        for idx in range(num_swp):
            fun = interp.interp1d(freq_log, diff_arr[idx, :], kind='cubic', copy=False,
                                  assume_sorted=True)
            f3db_list.append(10.0 ** (cls._get_intersect(fun, freq_log[0], freq_log_max[idx])))

        return gain_arr, np.array(f3db_list).reshape(gain_arr.shape)

    @classmethod
    def _compute_integrated_noise(cls, f_vec, out_density_arr, bw_arr, freq_idx):
        # move frequency axis to last axis
        out_density_arr = np.moveaxis(out_density_arr, freq_idx, -1)

        integ_noise_list = []
        for density, bw in zip(out_density_arr.reshape([-1, out_density_arr.shape[-1]]),
                               bw_arr.reshape([-1, bw_arr.shape[-1]])):
            noise_fun = interp.interp1d(f_vec, density**2, kind='cubic')
            integ_noise_list.append(integ.quad(noise_fun, f_vec[0], bw)[0])

        return np.sqrt(integ_noise_list)

    @classmethod
    def _compute_tset(cls, t, vout_arr, vin_arr, tot_err, gain_arr, time_idx, plot_flag=False):
        # move time axis to last axis
        vout_arr = np.moveaxis(vout_arr, time_idx, -1)
        vin_arr = np.moveaxis(vin_arr, time_idx, -1)*2
        tset_list = []
        if plot_flag:
            plt.figure()
            plt.plot(t, [1+tot_err]*len(t), 'r--')
            plt.plot(t, [1-tot_err]*len(t), 'r--')

        for vout, vin, gain in zip(vout_arr.reshape([-1, t.size]), vin_arr.reshape([-1, t.size]), gain_arr.flatten()):
            # since the evaluation of the raw data needs some of the constraints we need to do tset calculation here
            y = np.abs((vout - vout[0]) / gain / (vin[-1] - vin[0]))

            if plot_flag:
                plt.plot(t, y)
                plt.savefig('tset_debug.png', dpi=200)

            last_idx = np.where(y < 1.0 - tot_err)[0][-1]
            last_max_vec = np.where(y > 1.0 + tot_err)[0]
            if last_max_vec.size > 0 and last_max_vec[-1] > last_idx:
                last_idx = last_max_vec[-1]
                last_val = 1.0 + tot_err
            else:
                last_val = 1.0 - tot_err

            if last_idx == t.size - 1:
                return t[-1]
            f = interp.InterpolatedUnivariateSpline(t, y - last_val)
            t0 = t[last_idx]
            t1 = t[last_idx + 1]
            tset_list.append(cls._get_intersect(f, t0, t1))


        tset_arr = np.reshape(tset_list, newshape=gain_arr.shape)

        return tset_arr

    @classmethod
    def _get_intersect(cls, fun, xmin, xmax):
        try:
            return sciopt.brentq(fun, xmin, xmax)
        except ValueError:
            return np.NAN

class TIAMM(MeasurementManager):

    def __init__(self, *args, **kwargs):
        MeasurementManager.__init__(self, *args, **kwargs)

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'tia'

    def process_output(self, state, data, tb_manager):
        # type: (str, Dict[str, Any], TIAMM) -> Tuple[bool, str, Dict[str, Any]]

        done = True
        next_state = ''

        ax = plt.gca()
        self.add_plot(state, data, 'outpTIA', ax=ax, title='R_tia', function=lambda x: 20*np.log10(2*np.abs(x)), save=False)
        self.add_plot(state, data, 'outdiff', ax=ax, title='R_tia_ctle', function=lambda x: 20*np.log10(np.abs(x)), save=True)
        self.add_plot(state, data, 'input_noise', title='input_noise', function=lambda x:np.abs(x))
        self.add_plot(state, data, 'out_tran', x_axis='time', log_axis='none', title='out_tran', function=lambda x:np.abs(x))
        # self.add_plot(state, data, 'in_tran', x_axis='time', log_axis='none', title='in_tran')


        ac_res = tb_manager.get_R_and_f3db(data, ['outdiff'])
        tran_res = tb_manager.get_tset(data, out_name='out_tran', input_name='in_tran',
                                       tot_err=self.specs['tset_tol'], gain=ac_res['R_TIA_outdiff'])

        f3db_list = ac_res['f3db_outdiff']
        noise_res = tb_manager.get_integrated_noise(data, ['input_noise'], f3db_list)

        output = dict(
            r_tia=ac_res['R_TIA_outdiff'],
            f3db=f3db_list,
            rms_input_noise=noise_res['rms_input_noise'],
            ibias=np.abs(data['ibias']),
            tset=tran_res['tset_out_tran'],
        )

        return done, next_state, output

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
        combos = itertools.product(*(list(range(len(data[swp_kwrd]))) for swp_kwrd in sweep_kwrds))

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






