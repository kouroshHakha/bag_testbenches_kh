from bag.simulation.core import TestbenchManager
import scipy.interpolate as interp
import scipy.optimize as sciopt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import os

from typing import TYPE_CHECKING, Dict, List, Union, Any
if TYPE_CHECKING:
    from bag.core import Testbench


class NoiseWithCDFFitting(TestbenchManager):
    def setup_testbench(self, tb):
        # type: (Testbench) -> None
        sim_vars = self.specs['sim_vars']
        sim_outputs = self.specs.get('sim_outputs', None)

        for key, val in sim_vars.items():
            if isinstance(val, int) or isinstance(val, float):
                tb.set_parameter(key, val)
            else:
                tb.set_sweep_parameter(key,
                                       start=float(val[0]),
                                       stop=float(val[1]),
                                       step=float(val[2]))

        if sim_outputs:
            for key, val in sim_outputs.items():
                tb.add_output(key, val)

    def get_noise_offset(self, data, Tper, tdelay, fig_loc=None):
        # type : (Dict[str, Any], float, float) -> Dict[str, List[float]]

        sigma_list, offset_list = list(), list()
        axis_names = ['corner', 'vin', 'time']
        sweep_vars = data['sweep_params']['VO']
        swp_corner = ('corner' in sweep_vars)
        time = data['time']
        vin = data['vin']
        vo_corner = data['VO'].copy()

        if not swp_corner:
            corner_list = [self.env_list[0]]
            sweep_vars = ['corner'] + sweep_vars
            vo_corner = vo_corner[None, :]
        else:
            corner_list = data['corner'].tolist()

        order = [sweep_vars.index(swp) for swp in axis_names]
        vo_corner = np.transpose(vo_corner, axes=order)
        for corner, vo_arr in zip(corner_list, vo_corner):
            prob_one_list = list()
            for vo in vo_arr:
                fval = interp.interp1d(time, vo, kind='cubic')
                tsample = tdelay + Tper/2
                zero_counter = 0
                one_counter = 0
                while tsample < time[-1]:
                    vsample = fval(tsample)

                    if vsample > 1e-6:
                        one_counter += 1
                    elif vsample < 1e-6:
                        zero_counter += 1
                    else:
                        raise ValueError("encountered a zero during sampling")

                    tsample += Tper

                prob_one = one_counter / (one_counter+zero_counter)
                prob_one_list.append(prob_one)

            prob_one_array = np.array(prob_one_list)

            def cdf(x, sigma, mu):
                return 0.5 * (erf((x - mu) / (2 ** 0.5 * sigma)) + 1)

            (sigma, mu), fit_cov = sciopt.curve_fit(cdf, vin, prob_one_array)

            x_val = np.linspace(vin[0], vin[-1], 50)
            fitted_val = cdf(x_val, sigma, mu)
            # print("sigma_{}={}".format(corner, sigma))
            # print("offset_{}={}".format(corner, mu))

            if fig_loc:
                plt.plot(x_val, fitted_val, 'r--')
                plt.plot(vin, prob_one_array, "*-")
                plt.savefig(os.path.join(fig_loc, "Noise_CDF_{}.png".format(corner)), dpi=200)
                plt.close()
            sigma_list.append(float(sigma))
            offset_list.append(float(mu))

        noise_offset_params = dict(
            corner=corner_list,
            sigma=sigma_list,
            offset=offset_list,
        )

        return noise_offset_params


class OverDriveTB(TestbenchManager):
    def setup_testbench(self, tb):
        # not done properly, safer to make them equal
        sim_vars = self.specs.get('sim_vars', None)
        if sim_vars is not None:
            sim_vars['td'] = sim_vars['Tper']/4
        sim_outputs = self.specs.get('sim_outputs', None)

        for key, value in sim_vars.items():
            tb.set_parameter(key, value)
        if sim_outputs is not None:
            for key, val in sim_outputs.items():
                tb.add_output(key, val)

    def add_plot(self, data, yaxis_key=None, xaxis_key='time'):
        if yaxis_key is None:
            raise ValueError('yaxis_key should be specified')
        if yaxis_key not in data:
            raise ValueError('yaxis_key = {} not found in data keywords'.format(yaxis_key))
        plt.plot(data[xaxis_key], data[yaxis_key])

    def save_plot(self, fname):
        plt.grid()
        if os.path.isfile(fname):
            os.remove(fname)
        plt.savefig(fname, dpi=200)
        plt.close()

    def get_overdrive_params(self, data, Tper, tsetup, c_wait, fig_loc=None):
        v_charge_list, v_reset_list, v_out_list, ibias_list = list(), list(), list(), list()
        axis_names = ['corner', 'time']
        sweep_vars = data['sweep_params']['outdiff']
        swp_corner = ('corner' in sweep_vars)
        time = data['time']
        vout_corner = data['outdiff'].copy()

        if not swp_corner:
            corner_list = [self.env_list[0]]
            sweep_vars = ['corner'] + sweep_vars
            vout_corner = vout_corner[None, :]
        else:
            corner_list = data['corner'].tolist()

        order = [sweep_vars.index(swp) for swp in axis_names]
        vout_corner = np.transpose(vout_corner, axes=order)

        for corner, vout in zip(corner_list, vout_corner):

            fvout = interp.interp1d(time, vout, kind='cubic')
            t_charge = c_wait * Tper - tsetup
            t_reset = (c_wait+0.5) * Tper - tsetup
            t_out = (c_wait + 1) * Tper - tsetup

            v_charge = fvout(t_charge)
            v_reset = fvout(t_reset)
            v_out = fvout(t_out)

            index = corner_list.index(corner)

            if swp_corner:
                ibias = np.abs(data['ibias'][index])
            else:
                ibias = np.abs(data['ibias'])

            if fig_loc:
                if swp_corner:
                    plt.plot(time, data['inclk'][index])
                    plt.plot(time, data['outdiff'][index])
                    plt.plot(time, data['indiff'][index])
                else:
                    plt.plot(time, data['inclk'])
                    plt.plot(time, data['outdiff'])
                    plt.plot(time, data['indiff'])
                plt.savefig(os.path.join(fig_loc, 'overdrive_{}.png'.format(corner)), dpi=200)
                plt.close()

            v_charge_list.append(float(v_charge))
            v_reset_list.append(float(v_reset))
            v_out_list.append(float(v_out))
            ibias_list.append(float(ibias))

        output = dict(
            v_charge=v_charge_list,
            v_reset=v_reset_list,
            v_out=v_out_list,
            ibias=ibias_list,
            corner=corner_list,
        )

        return output