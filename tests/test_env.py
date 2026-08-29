from pathlib import Path

from pettingzoo.test import parallel_api_test

from biofoundry.config import load_config
from biofoundry.env import BioFoundryParallelEnv

ROOT = Path(__file__).resolve().parents[1]


def test_parallel_env_contract() -> None:
    env = BioFoundryParallelEnv(load_config(ROOT / "configs" / "demo.yaml"))
    parallel_api_test(env, num_cycles=25)
