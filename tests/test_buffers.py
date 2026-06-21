import torch

from rlens.core.buffers import RolloutBuffer, compute_gae


def test_gae_no_done_matches_manual():
    # single env, 3 steps, no episode boundaries
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])
    last_value = torch.tensor([0.5])
    last_done = torch.tensor([0.0])
    gamma, lam = 0.99, 0.95

    adv, ret = compute_gae(rewards, values, dones, last_value, last_done, gamma, lam)

    # manual backward recursion
    deltas = []
    for t in range(3):
        nv = last_value if t == 2 else values[t + 1]
        deltas.append(rewards[t] + gamma * nv - values[t])
    exp = torch.zeros(3, 1)
    gae = torch.zeros(1)
    for t in reversed(range(3)):
        gae = deltas[t] + gamma * lam * gae
        exp[t] = gae
    assert torch.allclose(adv, exp, atol=1e-6)
    assert torch.allclose(ret, adv + values, atol=1e-6)


def test_gae_done_cuts_bootstrap():
    # done at step 1 means step 0 must not bootstrap through step 1
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[10.0], [20.0]])
    dones = torch.tensor([[0.0], [1.0]])  # dones[1]=1 -> boundary before step 1
    last_value = torch.tensor([99.0])
    last_done = torch.tensor([0.0])
    adv, _ = compute_gae(rewards, values, dones, last_value, last_done, 0.99, 0.95)
    # step 0: next_nonterminal = 1 - dones[1] = 0, so delta = r0 - v0
    assert torch.allclose(adv[0], torch.tensor([1.0 - 10.0]), atol=1e-6)


def test_rollout_buffer_minibatches_cover_all():
    T, N, obs_dim = 8, 2, 4
    buf = RolloutBuffer(T, N, (obs_dim,), (), torch.device("cpu"))
    for _ in range(T):
        buf.add(
            torch.randn(N, obs_dim),
            torch.zeros(N),
            torch.zeros(N),
            torch.ones(N),
            torch.zeros(N),
            torch.zeros(N),
        )
    buf.compute_advantages(torch.zeros(N), torch.zeros(N), 0.99, 0.95)
    seen = 0
    for mb in buf.iter_minibatches(num_minibatches=4):
        seen += mb["obs"].shape[0]
        assert mb["obs"].shape[1] == obs_dim
    assert seen == T * N
