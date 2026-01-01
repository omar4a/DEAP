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
