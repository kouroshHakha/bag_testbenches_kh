# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design.module import Module
from bag.math import float_to_si_string


# noinspection PyPep8Naming
class bag_testbenches_kh__photonic_link_AFE_cm_tb(Module):
    """Module for library bag_testbenches_kh cell photonic_link_AFE_cm_tb.

    Fill in high level description here.
    """
    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'photonic_link_AFE_cm_tb.yaml'))


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
            self,
            dut_lib='',
            dut_cell='',
            dut_conns=None,
            vbias_dict=None,
            ibias_dict=None,
            no_cload=False,
            no_cpd=False,
    ):
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        if vbias_dict is None:
            vbias_dict = {}
        if ibias_dict is None:
            ibias_dict = {}
        if dut_conns is None:
            dut_conns = {}

        self.design_voltage_current_sources(vbias_dict, ibias_dict,
                                            v_inst_names=['VSUP'],
                                            i_inst_names=['IBIAS', 'Istream', 'Itran_in'])

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

    def design_voltage_current_sources(self, v_source_dict, i_source_dict,
                                       v_inst_names=None, i_inst_names=None):
        for source_inst_names, source_dict in ((i_inst_names, i_source_dict), (v_inst_names, v_source_dict)):
            for source_inst_name in source_inst_names:
                source_replacements_dict = source_dict.get(source_inst_name, {})

                if not source_replacements_dict:
                    self.delete_instance(source_inst_name)
                    continue

                assert isinstance(source_replacements_dict, dict), "{} is not a dictionary".format(source_inst_name)

                name_list, term_list, param_dict_list = [], [], []
                for inst_name, inst_properties in source_replacements_dict.items():
                    pname, nname = inst_properties[:2]
                    term_list.append(dict(PLUS=pname, MINUS=nname))
                    param_dict_list.append(inst_properties[2])
                    name_list.append(inst_name)

                self.array_instance(source_inst_name, name_list, term_list=term_list)

                for inst, inst_properties in zip(self.instances[source_inst_name],
                                                 source_replacements_dict.values()):
                    for k, v in inst_properties[2].items():
                        if isinstance(v, str):
                            pass
                        elif isinstance(v, int) or isinstance(v, float):
                            v = float_to_si_string(v)
                        else:
                            raise ValueError('value %s of type %s not supported' % (v, type(v)))
                        inst.parameters[k] = v
