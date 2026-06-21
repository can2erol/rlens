import json

import pytest
from typer.testing import CliRunner

from rlens.cli import app
from rlens.experiment.config import TrainConfig
from rlens.experiment.overrides import apply_overrides, parse_set

runner = CliRunner()


def test_parse_set():
    assert parse_set(["lr=3e-4", "hidden=[256,256]"]) == {"lr": "3e-4", "hidden": "[256,256]"}
    assert parse_set(None) == {}
    with pytest.raises(ValueError):
        parse_set(["bogus"])  # no '='


def test_algo_hyperparams_are_coerced_and_routed():
    cfg = TrainConfig(algo="ppo")
    apply_overrides(cfg, {"lr": "3e-4", "gamma": "0.95", "hidden": "[64,64]"})
    assert cfg.algo_overrides["lr"] == pytest.approx(3e-4)
    assert cfg.algo_overrides["gamma"] == pytest.approx(0.95)
    assert cfg.algo_overrides["hidden"] == (64, 64)
    assert isinstance(cfg.algo_overrides["hidden"], tuple)


def test_run_level_fields_set_directly():
    cfg = TrainConfig(algo="dqn")
    apply_overrides(cfg, {"num_envs": "4", "rollout_len": "256"})
    assert cfg.num_envs == 4
    assert cfg.rollout_len == 256
    assert "num_envs" not in cfg.algo_overrides  # routed to the dataclass field, not algo


def test_bool_coercion():
    cfg = TrainConfig(algo="sac")
    apply_overrides(cfg, {"autotune_alpha": "false"})
    assert cfg.algo_overrides["autotune_alpha"] is False


def test_unknown_key_raises_with_valid_keys():
    cfg = TrainConfig(algo="ppo")
    with pytest.raises(ValueError) as ei:
        apply_overrides(cfg, {"learning_rate": "0.1"})  # not a real field
    msg = str(ei.value)
    assert "unknown override key 'learning_rate'" in msg
    assert "lr" in msg  # suggests the valid keys


def test_cli_train_applies_overrides(tmp_path):
    result = runner.invoke(
        app,
        [
            "train",
            "--algo", "ppo",
            "--env", "CartPole-v1",
            "--steps", "1500",
            "--device", "cpu",
            "--num-envs", "4",
            "--runs-dir", str(tmp_path),
            "--name", "ovr",
            "--set", "lr=0.001",
            "--set", "hidden=[32,32]",
        ],
    )
    assert result.exit_code == 0, result.output
    meta = json.loads((tmp_path / "ovr" / "run.json").read_text())
    cfg = meta["config"]
    assert cfg["num_envs"] == 4
    assert cfg["algo_overrides"]["lr"] == pytest.approx(0.001)
    assert cfg["algo_overrides"]["hidden"] == [32, 32]  # tuple -> json list


def test_cli_unknown_set_key_fails(tmp_path):
    result = runner.invoke(
        app,
        ["train", "--algo", "ppo", "--device", "cpu", "--runs-dir", str(tmp_path),
         "--set", "nope=1"],
    )
    assert result.exit_code != 0
    assert "unknown override key" in result.output


def test_cli_requires_algo_without_config(tmp_path):
    result = runner.invoke(app, ["train", "--device", "cpu", "--runs-dir", str(tmp_path)])
    assert result.exit_code != 0
