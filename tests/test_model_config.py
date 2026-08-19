from vix_regime_allocation import model_config


def test_model_configuration_is_exact() -> None:
    assert model_config.SUPPORTED_STATE_COUNTS == (2, 3)
    assert model_config.HMM_SEEDS == (42, 43, 44, 45, 46)
    assert model_config.HMM_N_ITER == 500
    assert model_config.HMM_TOL == 1e-6
    assert model_config.HMM_MIN_COVAR == 1e-6
    assert model_config.STATIONARY_TOL == 1e-10
    assert model_config.PROBABILITY_TOL == 1e-8
    assert model_config.LIKELIHOOD_TIE_TOL == 1e-12
    assert model_config.BIC_TIE_TOL == 1e-12
    assert model_config.HMM_MIN_STATE_OCCUPANCY == 0.05
