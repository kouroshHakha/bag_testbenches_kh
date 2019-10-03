from bag.simulation.core import MeasurementManager

from typing import TYPE_CHECKING, Tuple, Dict, Any, Union
if TYPE_CHECKING:
    from .ComparatorTB import NoiseWithCDFFitting, OverDriveTB


class MyComparatorMM (MeasurementManager):
    def __init__(self, *args, **kwargs):
        MeasurementManager.__init__(self, *args, **kwargs)
        self.overall_results = dict()

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'od'

    def process_output(self, state, data, tb_manager):
        # type: (str, Dict[str, Any], Union[NoiseWithCDFFitting, OverDriveTB]) -> Tuple[bool, str, Dict[str, Any]]
        done = False
        next_state = ''

        if state == 'od':
            results = tb_manager.get_overdrive_params(data,
                                                      Tper=self.specs['testbenches']['od']['sim_vars']['Tper'],
                                                      tsetup=self.specs['tsetup'],
                                                      c_wait=self.specs['testbenches']['od']['sim_vars']['c_wait'],
                                                      fig_loc=self.data_dir)
            next_state = 'noise'
        elif state == 'noise':
            results = tb_manager.get_noise_offset(data,
                                                  Tper=self.specs['testbenches']['noise']['sim_vars']['Tper'],
                                                  tdelay=self.specs['testbenches']['noise']['sim_vars']['tdelay']/2,
                                                  fig_loc=self.data_dir)
            done = True

        else:
            raise ValueError('Unknown state: %s' % state)

        self.overall_results.update(**results)
        return done, next_state, self.overall_results
