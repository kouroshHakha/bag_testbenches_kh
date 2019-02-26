# -*- coding: utf-8 -*-

"""This module defines an extension of AC testbench class found in bag_testbenches_ec."""

import numpy as np
from scipy import interpolate
import IPython
import pdb
import matplotlib.pyplot as plt
import itertools
import os
from bag.io.sim_data import save_sim_results

from verification_ec.ac.core import ACTB



class GenericACTB(ACTB):
    @classmethod
    def get_dc_gain_max_gain_first_pole(cls, data, output_list):
        """
        Returns a dictionary of dc gain, maximum gain, and first pole
        This is makes sense if we have a CTLE where there is a LHP zero at low frequencies
        But if the frequency behaviour is monotonically decreasing it should be equivalent
        to get_gain_and_w3db(...)

        Parameters
        ----------
        data : Dict[str, Any]
            the simulation data dictionary.
        output_list : Sequence[str]
            list of output names to compute gain/bandwidth for.

        Returns
        -------
        output_dict : Dict[str, Any]
            A BAG data dictionary containing the gain/bandwidth information.
        """
        output_dict = {}
        swp_info = data['sweep_params']
        f_vec = data['freq']
        for out_name in output_list:
            out_arr = data[out_name]
            swp_params = swp_info[out_name]
            freq_idx = swp_params.index('freq')
            new_swp_params = [par for par in swp_params if par != 'freq']
            dc_gain, max_gain, first_pole = cls._compute_dc_gain_max_gain_first_pole(f_vec,np.abs(out_arr), freq_idx)


            cls.record_array(output_dict, data, dc_gain, 'dc_gain_' + out_name, new_swp_params)
            cls.record_array(output_dict, data, max_gain, 'max_gain_' + out_name, new_swp_params)
            cls.record_array(output_dict, data, first_pole, 'first_pole_' + out_name, new_swp_params)

        return output_dict

    @classmethod
    def _compute_dc_gain_max_gain_first_pole(cls, f_vec, out_arr, freq_idx):
        """
        General Idea, we have 3 cases:
            1. normal CTLE operation with a bump
            2. no zero at low frequency, just like an amplifier
            3. wierd behavior of going down and then comming back up
        We find the intersections with c and 0.99*dc_gain
        in case 1, 1.01*dc_gain is gonna happen before 0.99*dc_gain
        in case 2, 1.01*dc_gain has no crossings and 0.99*dc_gain has one
        in case 3, 0.99*dc_gain is gonna happen before 1.01*dc_gain
        In case 2,3 first pole is going to be w3db, but in case 3 it is 
        computed assuming 2nd pole onwards are far.
        
        Simpler solution is adopted for now
        
        Parmeters
        ---------
        f_vec : np.ndarray
            the frequency vector.  Must be sorted.
        out_arr : np.ndarray
            the block's output transfer function.  Could be multidimensional.
        freq_idx : int
            frequency axis index.

        Returns
        -------
        dc_gain : np.ndarray
            the DC gain array.
        max_gain : np.ndarray
            the maximum gain array.
        first_pole: np.ndarray
            the first pole array, it could be w3db if max_gain is dc_gain.
            If that's not the case the first pole is drived from the intersection
            of the theoretical line with transfer function if the remaining poles are
            assumed to be far
        """

        first_pole_list = []
        # move frequency axis to last axis
        out_arr = np.moveaxis(out_arr, freq_idx, -1)
        gain_arr = out_arr[..., 0]
        max_gain_arr = np.max(out_arr, axis=-1)

        output_shape = gain_arr.shape
        _, w3db_arr = cls._compute_gain_and_w3db(f_vec, out_arr, freq_idx)

        # gain_flat = gain_arr.flatten()
        # max_gain_flat = max_gain_arr.flatten()
        w3db_flat = w3db_arr.flatten()
        out_arr_flat = np.reshape(out_arr, newshape=(-1, out_arr.shape[-1]))

        for w3db, vout in zip(w3db_flat, out_arr_flat):
            dc_gain = vout[0]
            max_gain = np.max(vout)
            upper_bound_idx = np.argmax(vout)
 
            # fun_upper = interpolate.interp1d(f_vec, vout-1.01*dc_gain, kind='cubic')
            # fun_lower = interpolate.interp1d(f_vec, vout-0.99*dc_gain, kind='cubic')
            # upper_intersect = cls._get_intersect(fun_upper, f_vec[0], f_vec[upper_bound_idx])
            # lower_intersect = cls._get_intersect(fun_lower, f_vec[0], f_vec[-1])
            
            # rule of thumb: if there is at list 1% bump the behaviour is like CTLE otherwise it's just w3db
            if (max_gain / dc_gain) <= 1.01:
                first_pole_list.append(w3db)
            else:
                # compute the intersection with the theoretical line
                intersect = dc_gain / np.sqrt(2) * np.sqrt(1 + (max_gain / dc_gain) ** 2)
                fzero = interpolate.interp1d(f_vec, vout - intersect, kind='cubic')
                first_pole = cls._get_intersect(fzero, f_vec[0], f_vec[upper_bound_idx])
                first_pole_list.append(first_pole)

        first_pole_arr = np.reshape(first_pole_list, newshape=output_shape)
        return gain_arr, max_gain_arr, first_pole_arr