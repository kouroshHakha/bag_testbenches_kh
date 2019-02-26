# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design.module import Module


# noinspection PyPep8Naming
class bag_testbenches_kh__tia_noise_tb(Module):
    """Module for library bag_testbenches_kh cell tia_noise_tb.

    Fill in high level description here.
    """
    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'tia_noise_tb.yaml'))


    def __init__(self, database, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, self.yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
            dut_conns='DUT connection dictionary',
            vbias_dict='Vbias dictionary (could include VDD)',
            ibias_dict='Ibias dictionary (should include input current)',
            no_cload='if True cload is deleted',
            no_cpd='if True cpd is deleted',
        )
    def design(
            self,  # type: bag_testbenches_kh__tia_tb
            dut_lib='',  # type: str
            dut_cell='',  # type: str
            dut_conns=None,
            vbias_dict=None,
            ibias_dict=None,
            no_cload=False,
            no_cpd=False
            ):
        # type: (...) -> None

        if vbias_dict is None:
            vbias_dict = {}
        if ibias_dict is None:
            ibias_dict = {}
        if dut_conns is None:
            dut_conns = {}

        # setup bias sources
        self.design_dc_bias_sources(vbias_dict, ibias_dict, 'VSUP', 'IBIAS', define_vdd=True)

        # delete load cap if needed
        if no_cload:
            self.delete_instance('CLOAD')
        # delete input cap if needed
        if no_cload:
            self.delete_instance('CPD')

        # setup DUT
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)
        for term_name, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', term_name, net_name)

