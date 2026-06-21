import torch

from rlens.algos.dqn import DQN, DQNConfig
from rlens.core.env import EnvManager
from rlens.telemetry.recorder import Recorder
from rlens.trainer import Trainer

CPU = torch.device("cpu")


def _run(tmp_path, name, *, gradient_steps=1, update_every=1, num_envs=1, total_steps=400):
    env = EnvManager("CartPole-v1", num_envs=num_envs, seed=0)
    rec = Recorder(tmp_path / name)
    dqn = DQN(env, CPU, DQNConfig(batch_size=8))
    tr = Trainer(
        dqn, env, rec, CPU,
        total_steps=total_steps, update_every=update_every, gradient_steps=gradient_steps,
        learning_starts=20, progress=False,
    )
    tr.train()
    rec.close()
    env.close()
    return dqn.updates, tr.global_step


def test_gradient_steps_scales_update_count(tmp_path):
    """gradient_steps multiplies the number of gradient updates for the same collection."""
    u1, _ = _run(tmp_path, "g1", gradient_steps=1)
    u3, _ = _run(tmp_path, "g3", gradient_steps=3)
    assert u1 > 0
    assert u3 == 3 * u1  # same trigger count, 3x updates each


def test_update_every_reduces_updates(tmp_path):
    """A larger update_every means fewer training triggers -> fewer updates."""
    u_dense, _ = _run(tmp_path, "d", update_every=1)
    u_sparse, _ = _run(tmp_path, "s", update_every=4)
    assert u_sparse < u_dense
    assert abs(u_sparse - u_dense / 4) <= 2  # ~1/4 the updates


def test_vectorized_offpolicy_runs(tmp_path):
    """num_envs > 1 collection works for off-policy and reaches the step budget."""
    updates, steps = _run(tmp_path, "vec", num_envs=4, total_steps=400)
    assert steps >= 400
    assert updates > 0
