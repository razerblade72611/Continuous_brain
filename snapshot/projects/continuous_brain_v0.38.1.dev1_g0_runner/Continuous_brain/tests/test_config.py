from continuous_brain_snn.config import E0Config, small_config


def test_config_hash_is_stable_and_sensitive():
    assert E0Config().config_hash() == E0Config().config_hash()
    c = E0Config()
    c.neuron.noise_std += 0.01
    assert c.config_hash() != E0Config().config_hash()


def test_round_trip():
    c = small_config(5)
    assert E0Config.from_dict(c.to_dict()).config_hash() == c.config_hash()
