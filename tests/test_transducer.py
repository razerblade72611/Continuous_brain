import pytest
import torch

from continuous_brain.transducer import (FeatureAssembler, LatentTokenPool, MultiscaleTraceBank,
                                         PopulationAssignment, PopulationPooling, SpikeEvents,
                                         population_synchrony)

BIN = 0.001
TAUS = (0.005, 0.025, 0.1, 0.5, 2.0)


@pytest.mark.parametrize("t_start", [0.0, 3 * 86400.0 + 0.123])
def test_parallel_traces_match_sequential_recurrence(t_start):
    b, n, t_bins = 2, 30, 20
    bank = MultiscaleTraceBank(n, b, TAUS, BIN, 0.05)
    bank.reset(t_start)
    ref = torch.zeros(b, n, len(TAUS))
    decay = torch.exp(-BIN / torch.tensor(TAUS))
    t = t_start
    for _ in range(40):
        raster = (torch.rand(b, t_bins, n) < 0.1).float()
        window = bank.advance(SpikeEvents.from_dense(raster, t, BIN), t + t_bins * BIN)
        for k in range(t_bins):
            ref = ref * decay + raster[:, k, :, None]
        t += t_bins * BIN
        assert torch.allclose(window.state, ref, atol=1e-4)


def test_event_times_are_float64():
    ev = SpikeEvents.from_dense(torch.ones(1, 1, 1), 86400.0, BIN)
    assert ev.time.dtype == torch.float64
    assert float(ev.time[0]) == pytest.approx(86400.001, abs=1e-9)


def test_bin_edges_belong_to_their_bin():
    ev = SpikeEvents(torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0.001, 0.0015]),
                     torch.ones(2), 1, 1)
    dense = ev.bin(0.0, 3, BIN)
    assert dense[0, :, 0].tolist() == [1.0, 1.0, 0.0]


def test_events_outside_window_raise():
    ev = SpikeEvents(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.5]), torch.ones(1), 1, 1)
    with pytest.raises(ValueError):
        ev.bin(0.0, 10, BIN)


def test_invalid_indices_raise():
    with pytest.raises(ValueError):
        SpikeEvents(torch.tensor([1]), torch.tensor([0]), torch.tensor([0.0]), torch.ones(1), 1, 1)


def test_recency_and_never_spiked():
    bank = MultiscaleTraceBank(2, 1, TAUS, BIN, recency_tau_s=0.05)
    raster = torch.zeros(1, 20, 2)
    raster[0, 9, 0] = 1                      # neuron 0 spikes at t = 0.010
    window = bank.advance(SpikeEvents.from_dense(raster, 0.0, BIN), 0.020)
    assert float(window.recency[0, 0]) == pytest.approx(torch.exp(torch.tensor(-0.010 / 0.05)).item(), rel=1e-5)
    assert float(window.recency[0, 1]) == 0.0


def test_synchrony_separates_synchronous_from_asynchronous():
    n = 40
    assign = PopulationAssignment(torch.tensor([0] * 20 + [1] * 20), 2)
    bank = MultiscaleTraceBank(n, 1, TAUS, BIN, 0.05)
    raster = torch.zeros(1, 50, n)
    for k in range(0, 50, 10):
        raster[0, k, :20] = 1                               # population 0 fires in lockstep
    for i in range(20, 40):
        raster[0, (i * 7) % 50, i] = 1                      # population 1 staggered
    window = bank.advance(SpikeEvents.from_dense(raster, 0.0, BIN), 0.050)
    chi = population_synchrony(window.history_fast, assign)[0]
    assert chi[0] > 0.95 and chi[1] < 0.3


def test_burst_channel_and_derivatives():
    bank = MultiscaleTraceBank(1, 1, TAUS, BIN, 0.05)
    fa = FeatureAssembler(len(TAUS))
    single = torch.zeros(1, 20, 1)
    single[0, 19, 0] = 1
    f1 = fa(bank.advance(SpikeEvents.from_dense(single, 0.0, BIN), 0.020))
    burst = torch.zeros(1, 20, 1)
    burst[0, 17:20, 0] = 1
    f2 = fa(bank.advance(SpikeEvents.from_dense(burst, 0.020, BIN), 0.040))
    assert f1.shape[-1] == fa.n_channels == 3 * len(TAUS) + 2
    assert float(f1[0, 0, -1]) == 0.0          # one spike is not a burst
    assert float(f2[0, 0, -1]) > 0.0           # three within 3 ms is
    assert float(f2[0, 0, len(TAUS)]) > 0.0    # fast trace rose: dX > 0


def test_population_mean_pooling_is_exact():
    assign = PopulationAssignment(torch.tensor([0, 0, 1, 1, 1]), 2)
    x = torch.arange(5, dtype=torch.float32)[None]
    assert assign(x).tolist() == [[0.5, 3.0]]


def test_empty_population_rejected():
    with pytest.raises(ValueError):
        PopulationAssignment(torch.tensor([0, 0, 2]), 3)


def test_from_regions_uses_labels():
    assign = PopulationAssignment.from_regions(["v1", "pfc", "v1"])
    assert assign.n_populations == 2
    assert assign.sizes.tolist() == [1, 2]


def test_latent_pool_shapes_and_amplitude_sensitivity():
    pool = LatentTokenPool(in_channels=6, n_populations=4, n_tokens=3, dim=32, n_heads=4).eval()
    x = torch.randn(2, 4, 6)
    z = pool(x)
    assert z.shape == (2, 3, 32)
    # Same direction, larger amplitude must not map to the same latents.
    assert not torch.allclose(pool(x), pool(3.0 * x), atol=1e-4)


def test_pooling_pipeline_shapes():
    n, p = 32, 4
    bank = MultiscaleTraceBank(n, 1, TAUS, BIN, 0.05)
    fa = FeatureAssembler(len(TAUS))
    pp = PopulationPooling(PopulationAssignment.balanced(n, p))
    window = bank.advance(SpikeEvents.from_dense((torch.rand(1, 20, n) < 0.2).float(), 0.0, BIN), 0.02)
    pooled = pp(fa(window), window)
    assert pooled.shape == (1, p, pp.n_out_channels(fa.n_channels))
