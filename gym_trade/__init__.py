from gym.envs.registration import register
import logging


logger = logging.getLogger(__name__)

register(
    id='trade-v0',
    entry_point='gym_trade.envs:TRADEEnv',
)
