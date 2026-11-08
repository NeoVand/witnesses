"""Do a trained Newton layer's phases sit on the characters of (Z_3)^4? numpy only.

Each channel's feature on card a is f_c(a) = g_c(a) exp(i theta_c(a)), computed from the token features the model sees
(table cards carry no flag; the card in hand carries flag 62), with theta = 2 (h W_th + b_th), g = sigmoid(h W_g + b_g)
and h = tanh(x W_in), as in hoa_polybench.attention (phase 2). Its Fourier coefficients on (Z_3)^4 are
F_c(s) = mean_a f_c(a) conj(chi_s(a)), chi_s(a) = w^(s . a), w = exp(2 pi i / 3).

Channel share: the largest |F_c(s)|^2 over s != 0, divided by sum_s |F_c(s)|^2 (s = 0 included).
Run share: the mean of the channel shares over the 64 channels.
Found: the number of the 40 classes {s, -s}, s != 0, that are the top character of some channel with share >= 0.5.
Baseline: the same run share with every theta_c(a) replaced by an independent uniform phase, gates kept (200 draws).

python3 -I newton_chars.py --selftest
"""
import sys
import numpy as np

CARDS = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])
CHI = np.exp(2j * np.pi / 3 * ((CARDS @ CARDS.T) % 3))  # CHI[a, s] = w^(s . a); s indexed like the cards, s = 0 first
CODE = np.array([1, 3, 9, 27])
NEG = ((-CARDS) % 3) @ CODE                              # index of -s
CLASS = np.minimum(np.arange(81), NEG)                   # one label per conjugate pair; 40 labels for s != 0
PHASE = 2.0


def features(query):
    x = np.zeros((81, 64), np.float32)
    for a in range(4):
        x[np.arange(81), 3 * a + CARDS[:, a]] = 1
    if query:
        x[:, 62] = 1
    return x


def channels(P, query):
    """P: input (64, D), th_w (D, R), th_b (R,), g_w (D, R), g_b (R,). Returns gates and phases, each (81, R)."""
    h = np.tanh(features(query).astype(np.float64) @ P["input"])
    theta = PHASE * (h @ P["th_w"] + P["th_b"])
    g = 1 / (1 + np.exp(-(h @ P["g_w"] + P["g_b"])))
    return g, theta


def shares(g, theta):
    f = g * np.exp(1j * theta)
    power = np.abs(CHI.conj().T @ f / 81) ** 2           # (81 characters, R channels)
    nonzero = power[1:]
    return nonzero.max(0) / power.sum(0), nonzero.argmax(0) + 1


def analyse(P, rng, draws=200):
    out = {}
    for side, query in (("table", False), ("hand", True)):
        g, theta = channels(P, query)
        s, best = shares(g, theta)
        found = sorted({int(CLASS[b]) for b, v in zip(best, s) if v >= 0.5})
        base = np.array([shares(g, rng.uniform(0, 2 * np.pi, theta.shape))[0].mean() for _ in range(draws)])
        out[side] = dict(share=float(s.mean()), found=len(found), base_mean=float(base.mean()), base_max=float(base.max()),
                         channel_shares=[round(float(v), 4) for v in s], top=[CARDS[b].tolist() for b in best])
    return out


def from_legacy(get):
    """get(name) returns the array for one fit; names as in bench4.init's pytree."""
    return dict(input=get("['input']"), th_w=get("['legacy']['th']['w']"), th_b=get("['legacy']['th']['b']"),
                g_w=get("['legacy']['g']['w']"), g_b=get("['legacy']['g']['b']"))


def selftest():
    assert len(set(CLASS[1:].tolist())) == 40 and CLASS[0] == 0
    rng = np.random.default_rng(0)
    s_true = CARDS[rng.integers(1, 81, 64)]
    theta = 2 * np.pi / 3 * (CARDS @ s_true.T) + rng.uniform(0, 2 * np.pi, 64)   # exact characters, arbitrary offsets
    exact, best = shares(np.ones((81, 64)), theta)
    assert np.allclose(exact, 1) and ((CARDS[best] - s_true) % 3 == 0).all()
    noise = shares(np.ones((81, 64)), rng.uniform(0, 2 * np.pi, (81, 64)))[0].mean()
    half = shares(np.ones((81, 64)), theta + rng.normal(0, 0.6, (81, 64)))[0].mean()
    print("exact characters", round(float(exact.mean()), 4), "phase noise sd 0.6", round(float(half), 4),
          "uniform phases", round(float(noise), 4))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
