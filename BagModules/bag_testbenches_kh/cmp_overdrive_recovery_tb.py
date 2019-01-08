# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'cmp_overdrive_recovery_tb.yaml'))


# noinspection PyPep8Naming
class bag_testbenches_kh__cmp_overdrive_recovery_tb(Module):
    """
    Module for Overdrive Recovery test for comparators.
    TO use it for different architectures a wrapper should be built (i.e. DTSA_dsn_wrapper.py)
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
        )

    def design(
            self,  # type: bag_testbenches_kh__cmp_overdrive_recovery_tb
            dut_lib='',  # type: str
            dut_cell='',  # type: str
            ):
        # type: (...) -> None
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)