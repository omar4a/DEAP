import numpy as np

from deap_emotion.features import extract_features


def test_bandpower_peak_alpha():
    sfreq = 128
    t = np.arange(0, 2.0, 1.0 / sfreq)
    signal = np.sin(2 * np.pi * 10.0 * t)
    eeg = signal[None, None, :]

    bands = (
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 45.0),
    )
    features, _ = extract_features(eeg, sfreq, bands, mode="bandpower_log")
    assert features.shape == (1, len(bands))
    alpha_idx = 2
    assert features[0, alpha_idx] == features.max()


def test_feature_set_shapes():
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((1, 2, 256))
    bands = (("low", 1.0, 4.0), ("high", 4.0, 8.0))
    psd_bins = (("1-3", 1.0, 3.0), ("3-5", 3.0, 5.0))
    features, _ = extract_features(
        eeg,
        sfreq=128,
        bands=bands,
        mode="bandpower_log",
        feature_set=("bandpower", "psd", "de", "asymmetry"),
        psd_bands=psd_bins,
        asymmetry_pairs=[(0, 1)],
    )
    # bandpower: 2 ch × 2 bands = 4
    # psd: 2 ch × 2 bins = 4
    # de: 2 ch × 2 bands = 4
    # asymmetry: 1 pair × 2 bands × 2 (diff + ratio) = 4
    expected = (2 * 2) + (2 * 2) + (2 * 2) + (1 * 2 * 2)
    assert features.shape == (1, expected)
