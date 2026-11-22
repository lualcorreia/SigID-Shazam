"""Gera sinais sintéticos clássicos e mostra o que o modelo responde (teste de generalização)."""
import numpy as np
from scipy import signal as sps
from sigid_core import Identifier
import sigfeatures as sf_

FS = sf_.FS
rng = np.random.default_rng(7)
T = 8.0
t = np.arange(int(T * FS)) / FS


def noise(x, snr_db=15):
    p = np.mean(x ** 2)
    b, a = sps.butter(4, [250, 2900], btype="band", fs=FS)
    n = sps.lfilter(b, a, rng.normal(0, 1, len(x)))
    return x + n * np.sqrt(p / np.mean(n ** 2) / 10 ** (snr_db / 10))


def fsk(baud, f0, f1, bits=None):
    sps_ = FS / baud
    nb = int(len(t) / sps_) + 1
    bits = rng.integers(0, 2, nb) if bits is None else bits
    f = np.where(bits[(np.arange(len(t)) / sps_).astype(int)] == 1, f1, f0)
    return np.sin(2 * np.pi * np.cumsum(f) / FS)


def cw(wpm=20, f=700):
    dot = 1.2 / wpm
    key = np.zeros(len(t))
    pos = 0.0
    msg = "-.-. --.-  -.-. --.-  -.. .  .--. .--. ..--- .-.. .-  -.-"
    while pos < T:
        for ch in msg:
            n = {".": 1, "-": 3, " ": 0}[ch]
            if n:
                a, b = int(pos * FS), int((pos + n * dot) * FS)
                key[a:b] = 1
                pos += (n + 1) * dot
            else:
                pos += 2 * dot
        pos += 7 * dot
    key = np.convolve(key, np.hanning(int(0.005 * FS)), "same")
    return key / key.max() * np.sin(2 * np.pi * f * t)


def bpsk31(f=1000):
    baud = 31.25
    n = int(T * baud) + 1
    sym = np.cumprod(np.where(rng.integers(0, 2, n) == 0, -1, 1))
    k = (np.arange(len(t)) * baud / FS)
    i = k.astype(int)
    frac = k - i
    nxt = np.append(sym[1:], sym[-1])
    env = np.where(sym[i] == nxt[np.minimum(i, n - 1)], sym[i],
                   sym[i] * np.cos(np.pi * frac))  # transição em cosseno
    return env * np.sin(2 * np.pi * f * t)


tests = {
    "CW 20 WPM": cw(),
    "RTTY 45.45 Bd, shift 170 Hz": fsk(45.45, 2125, 2295),
    "AFSK 1200 Bd (Bell 202 / APRS)": fsk(1200, 1200, 2200),
    "BPSK31": bpsk31(),
    "Portadora (tom puro)": np.sin(2 * np.pi * 1000 * t),
    "Ruído de receptor SSB": np.zeros(len(t)),
}

ident = Identifier()
for name, x in tests.items():
    x = noise(x) if x.any() else noise(np.ones(len(t)) * 1e-6, -40)
    p, pm, _ = ident.identify_array(x.astype(np.float32), FS)
    mods = ", ".join(f"{m} {v:.0%}" for m, v in ident.top_mod(pm))
    print(f"\n{name}   [modulação: {mods}]")
    for i, (lab, v) in enumerate(ident.top(p, 5), 1):
        print(f"   {i}. {lab:55s} {v:5.1%}")
