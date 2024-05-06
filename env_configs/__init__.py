from .dqn_atari_config import atari_dqn_config
from .dqn_car_racing_config import car_racing_dqn_config
from .dqn_basic_config import basic_dqn_config
from .sac_config import sac_config

configs = {
    "dqn_atari": atari_dqn_config,
    "dqn_car_racing": car_racing_dqn_config,
    "dqn_basic": basic_dqn_config,
    "sac": sac_config,
}
