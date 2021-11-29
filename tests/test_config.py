import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from iiif.config import Config, load_config


def test_load_with_env_var(tmp_path: Path, config: Config):
    config_path = tmp_path / 'config.yml'
    with config_path.open('w') as f:
        yaml.dump(config.raw, f)
    with patch.dict(os.environ, {'IIIF_CONFIG': str(config_path)}):
        loaded_config = load_config()
    assert loaded_config.raw == config.raw


def test_load_with_path(config: Config):
    with pytest.raises(Exception, match='IIIF_CONFIG'):
        load_config()


def test_load_missing_path():
    with pytest.raises(Exception, match='does not exist'):
        with patch.dict(os.environ, {'IIIF_CONFIG': '/dev/null/missing'}):
            load_config()


def test_has_default_profile(tmp_path: Path, config: Config):
    config_path = tmp_path / 'config.yml'
    with config_path.open('w') as f:
        yaml.dump(config.raw, f)
    with patch.dict(os.environ, {'IIIF_CONFIG': str(config_path)}):
        loaded_config = load_config()
    assert loaded_config.has_default_profile()
