# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design.module import Module


# noinspection PyPep8Naming
class bag_testbenches_kh__photonic_link_AFE_CMP_wrapper(Module):
    """Module for library bag_testbenches_kh cell photonic_link_AFE_CMP_wrapper.

    Fill in high level description here.
    """
    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'photonic_link_AFE_CMP_wrapper.yaml'))


    def __init__(self, database, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, self.yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            dut_lib='Device-under-test library name.',
            dut_cell='Device-under-test cell name.',
            dut_conns='DUT connection dictionary.',
        )

    def design(self,
               dut_lib='',  # type: str
               dut_cell='',  # type: str
               dut_conns=None,  # type: Dict[str, str]
               ):

        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)

        # if dut_conns are different from the default, reconnect the terminals
        for dut_pin, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', dut_pin, net_name)

