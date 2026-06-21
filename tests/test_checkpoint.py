import numpy as np
import torch
from typer.testing import CliRunner

from rlens.algos.dqn import DQN, DQNConfig
from rlens.algos.sac import SAC, SACConfig
from rlens.cli import app
from rlens.core.env import EnvManager
from rlens.experiment.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from rlens.experiment.run import resume_training, train_single
from rlens.telemetry.recorder import Recorder
from rlens.telemetry.store import TelemetryStore, read_meta

runner = CliRunner()
CPU = torch.device("cpu")


def _fill_and_update_dqn(dqn, rec, n_obs=60, n_upd=10):
    for _ in range(n_obs):
        dqn.observe(
            {
                "obs": np.random.randn(1, 4).astype(np.float32),
                "action": np.array([0], dtype=np.int64),
                "reward": np.array([1.0], dtype=np.float32),
                "next_obs": np.random.randn(1, 4).astype(np.float32),
                "done": np.array([0.0], dtype=np.float32),
            }
        )
    for i in range(n_upd):
        dqn.update_off_policy(rec, step=i)


def test_dqn_checkpoint_roundtrip(tmp_path):
    """A DQN's weights, target net, optimizer and counters survive a save/load cycle."""
    env = EnvManager("CartPole-v1", num_envs=1, seed=0)
    rec = Recorder(tmp_path / "rec")
    dqn = DQN(env, CPU, DQNConfig(batch_size=8))
    _fill_and_update_dqn(dqn, rec)
    env.close()
    rec.close()
    assert dqn.t == 60 and dqn.updates == 10

    path = save_checkpoint(tmp_path / "run", dqn, global_step=1234, config={})

    env2 = EnvManager("CartPole-v1", num_envs=1, seed=0)
    fresh = DQN(env2, CPU, DQNConfig(batch_size=8))
    env2.close()
    # fresh net starts different
    assert not all(
        torch.equal(a, b)
        for a, b in zip(dqn.q.parameters(), fresh.q.parameters(), strict=True)
    )

    ckpt = load_checkpoint(path, map_location=CPU)
    assert ckpt["global_step"] == 1234
    fresh.load_checkpoint_state(ckpt["algo"])

    assert fresh.t == 60 and fresh.updates == 10
    for a, b in zip(dqn.q.parameters(), fresh.q.parameters(), strict=True):
        assert torch.equal(a, b)
    for a, b in zip(dqn.q_target.parameters(), fresh.q_target.parameters(), strict=True):
        assert torch.equal(a, b)
    # optimizer momentum state was restored (non-empty)
    assert len(fresh.opt.state_dict()["state"]) > 0


def test_sac_checkpoint_restores_alpha(tmp_path):
    """SAC's autotuned temperature (log_alpha + its optimizer) round-trips intact."""
    env = EnvManager("Pendulum-v1", num_envs=1, seed=0)
    rec = Recorder(tmp_path / "rec")
    sac = SAC(env, CPU, SACConfig(batch_size=8))
    for _ in range(40):
        sac.observe(
            {
                "obs": np.random.randn(1, 3).astype(np.float32),
                "action": np.random.uniform(-2, 2, (1, 1)).astype(np.float32),
                "reward": np.array([-1.0], dtype=np.float32),
                "next_obs": np.random.randn(1, 3).astype(np.float32),
                "done": np.array([0.0], dtype=np.float32),
            }
        )
    for i in range(8):
        sac.update_off_policy(rec, step=i)
    env.close()
    rec.close()

    path = save_checkpoint(tmp_path / "run", sac, global_step=99, config={})

    env2 = EnvManager("Pendulum-v1", num_envs=1, seed=0)
    fresh = SAC(env2, CPU, SACConfig(batch_size=8))
    env2.close()
    fresh.load_checkpoint_state(load_checkpoint(path, map_location=CPU)["algo"])

    assert torch.equal(fresh.log_alpha, sac.log_alpha)
    assert fresh.alpha == sac.alpha
    assert fresh.updates == sac.updates
    # alpha optimizer still points at the live log_alpha tensor -> another step works
    fresh.update_off_policy(Recorder(tmp_path / "rec2"), step=100)


def test_find_latest_and_prune(tmp_path):
    env = EnvManager("CartPole-v1", num_envs=1, seed=0)
    dqn = DQN(env, CPU, DQNConfig(batch_size=8))
    env.close()
    for step in (10, 20, 30, 40):
        save_checkpoint(tmp_path / "run", dqn, step, config={}, keep_last=2)
    files = list((tmp_path / "run" / "checkpoints").glob("step_*.pt"))
    assert len(files) == 2
    assert find_latest_checkpoint(tmp_path / "run").name == "step_000000040.pt"


def test_resume_continues_training(tmp_path):
    """resume_training picks up from the checkpoint and extends to a higher step target."""
    run = train_single(
        algo="ppo",
        env_id="CartPole-v1",
        total_steps=2000,
        seed=0,
        device="cpu",
        runs_dir=tmp_path,
        name="resume-me",
        progress=False,
    )
    meta1 = read_meta(run)
    first_step = meta1["final_step"]
    assert first_step >= 2000
    assert find_latest_checkpoint(run) is not None

    store = TelemetryStore(run)
    n_before = len(store.episodes())
    store.close()

    run2 = resume_training(run, total_steps=4000, device="cpu", progress=False)
    assert run2 == run  # same dir, appended

    meta2 = read_meta(run)
    assert meta2["final_step"] >= 4000
    assert meta2["final_step"] > first_step

    store = TelemetryStore(run)
    n_after = len(store.episodes())
    store.close()
    assert n_after > n_before


def test_cli_resume(tmp_path):
    r1 = runner.invoke(
        app,
        ["train", "--algo", "ppo", "--steps", "2000", "--device", "cpu",
         "--runs-dir", str(tmp_path), "--name", "cli-resume"],
    )
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(
        app, ["train", "--resume", str(tmp_path / "cli-resume"), "--steps", "3500",
              "--device", "cpu"]
    )
    assert r2.exit_code == 0, r2.output
    assert read_meta(tmp_path / "cli-resume")["final_step"] >= 3500
