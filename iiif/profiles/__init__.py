from concurrent.futures import Executor

from copy import deepcopy
from typing import Dict

from iiif.config import Config
from iiif.profiles.base import AbstractProfile
from iiif.profiles.disk import OnDiskProfile
from iiif.profiles.mss import MSSProfile

registry = {
    'disk': OnDiskProfile,
    'mss': MSSProfile,
}


def load_profiles(config: Config, pool: Executor) -> Dict[str, AbstractProfile]:
    """
    Given the config object, create all the profiles that are defined within it and return them.

    :param config: the config object
    :param pool: pool for offloaded processing
    :return: a dict of profiles keyed by name
    """
    profiles = {}
    # use a deep copy of the profiles config so that we don't mess with the one stored on the
    # config object (we do a pop in the loop!)
    profile_options = deepcopy(config.profile_options)
    for name, options in profile_options.items():
        # here's that pop, wow!
        profile_creator = registry[options.pop('type')]
        rights = options.pop('rights')
        profiles[name] = profile_creator(name, config, pool, rights, **options)
    return profiles
