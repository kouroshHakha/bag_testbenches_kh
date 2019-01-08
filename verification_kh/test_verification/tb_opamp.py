"""
The script for testing the Design Manager Module
This file can generate layout/schematic, Do LVS and RCX, and run overdrive test recovery testbench
To be able to use this the top level yaml file has to follow certain conventions. DTSA.yaml is an example
"""
from bag import BagProject
from bag.simulation.core import DesignManager
from bag.io import read_yaml, open_file


if __name__ == '__main__':
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    fname = 'specs_design/opamp_two_stage_1e8.yaml'
    sim = DesignManager(bprj, fname)
    sim.characterize_designs(generate=False, measure=True, load_from_file=False)
