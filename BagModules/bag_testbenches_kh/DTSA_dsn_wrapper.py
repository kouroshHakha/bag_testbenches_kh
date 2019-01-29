# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'DTSA_dsn_wrapper.yaml'))


# noinspection PyPep8Naming
class bag_testbenches_kh__DTSA_dsn_wrapper(Module):
    """Module for library bag_testbenches_kh cell DTSA_dsn_wrapper.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
            dut_conns='DUT connection dictionary.',
        )

    def design(self,  # type: bag_testbenches_kh__DTSA_dsn_wrapper
               dut_lib='',  # type: str
               dut_cell='',  # type: str
               dut_conns=None,  # type: Dict[str, str]
               ):
        # type: (...) -> None
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)

        # if dut_conns are different from the default, reconnect the terminals
        for dut_pin, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', dut_pin, net_name)