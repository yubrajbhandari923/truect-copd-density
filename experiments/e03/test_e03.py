import torch

from ddpm3d import sample, scheduler


def test_ddim_sampler_recovers_x0_with_oracle_eps():
    """Full reverse sampler + sliding-window glue: with the exact epsilon, DDIM (eta 0) must return x0."""
    acp = scheduler().alphas_cumprod
    x0 = torch.rand(1, 1, 20, 24, 28) * 2 - 1

    def oracle(inp, t):  # channel 0 carries x0 here (the CT slot), channel 1 is x_t
        a = acp[t]
        return (inp[:, 1:] - a.sqrt() * inp[:, :1]) / (1 - a).sqrt()

    for steps in (5, 25):
        x = sample(oracle, x0, steps, seed=0, roi=(16, 16, 16), sw_batch=2)
        assert (x - x0).abs().max() < 1e-3, (steps, (x - x0).abs().max())
