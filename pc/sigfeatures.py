"""
Extração de features de "inteligência de sinal" a partir de áudio demodulado.

A ideia é descrever o sinal por propriedades físicas que distinguem modos:
  * forma espectral (mel/MFCC, banda ocupada, planicidade)
  * estrutura de tons (picos, espaçamento entre tons -> shift FSK, OFDM)
  * histograma da frequência dominante relativo à mediana (invariante a sintonia)
  * linhas espectrais do envelope^2 e da frequência instantânea (taxa de baud)
  * potências do sinal analítico a^2, a^4, a^8 (assinatura BPSK/QPSK/8PSK)
  * envelope constante vs. variável (FSK/FM vs. AM/OFDM)
  * estrutura temporal lenta (ciclo de trabalho, keying CW, pulsos, varreduras)
"""
from fractions import Fraction
import numpy as np
import soundfile as sf
from scipy import signal as sps
from scipy.fft import rfft, rfftfreq, fft, ifft, dct

FS = 16000          # taxa interna de análise
WIN_S = 4.0         # janela de análise (s)
FEATURE_VERSION = 3
EPS = 1e-12
NFFT = 1024
HOP = 256


# ---------------------------------------------------------------- áudio
def to_mono_fs(x, sr, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if int(sr) != fs:
        fr = Fraction(fs, int(sr)).limit_denominator(2000)
        x = sps.resample_poly(x, fr.numerator, fr.denominator)
    return x.astype(np.float32)


def load_audio(path, fs=FS, max_s=180.0):
    info = sf.info(path)
    if info.samplerate > 96000:
        raise ValueError(f"taxa {info.samplerate} Hz: parece IQ/baseband, não áudio demodulado")
    frames = int(min(info.frames if info.frames > 0 else 1e12, max_s * info.samplerate))
    x, sr = sf.read(path, frames=frames, dtype="float32", always_2d=True)
    return to_mono_fs(x, sr, fs)


def segments(x, fs=FS, win_s=WIN_S, hop_s=2.0, max_n=30, keep_db=12.0):
    """Corta em janelas e mantém só as que têm energia perto da janela mais forte."""
    n = int(win_s * fs)
    if len(x) <= n:
        return [x]
    hop = int(hop_s * fs)
    starts = list(range(0, len(x) - n + 1, hop))
    segs = [x[s:s + n] for s in starts]
    rms = np.array([10 * np.log10(np.mean(s.astype(np.float64) ** 2) + EPS) for s in segs])
    keep = [s for s, r in zip(segs, rms) if r >= rms.max() - keep_db]
    if len(keep) > max_n:
        idx = np.linspace(0, len(keep) - 1, max_n).round().astype(int)
        keep = [keep[i] for i in idx]
    return keep


# ---------------------------------------------------------------- utilidades
def _mel(f):
    return 2595 * np.log10(1 + f / 700.0)


def _imel(m):
    return 700 * (10 ** (m / 2595.0) - 1)


def _mel_fb(n_mels=64, nfft=NFFT, fs=FS, fmin=30.0):
    freqs = rfftfreq(nfft, 1 / fs)
    pts = _imel(np.linspace(_mel(fmin), _mel(fs / 2), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        l, c, r = pts[i:i + 3]
        up = (freqs - l) / (c - l)
        down = (r - freqs) / (r - c)
        fb[i] = np.maximum(0, np.minimum(up, down))
        if fb[i].sum() == 0:
            fb[i, np.argmin(np.abs(freqs - c))] = 1.0
    return fb / fb.sum(1, keepdims=True)


_FB = _mel_fb()
_F = rfftfreq(NFFT, 1 / FS)
_DF = _F[1]


def _log_pool_max(spec, freqs, fmin, fmax, nb):
    edges = np.geomspace(fmin, fmax, nb + 1)
    idx = np.searchsorted(freqs, edges)
    out = np.zeros(nb)
    for i in range(nb):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        if a < len(spec):
            out[i] = spec[a:b].max()
    return out


def _line_spectrum_feats(sig, fs, fmin=2.0, fmax=4000.0, nb=40):
    """Espectro de um sinal auxiliar (envelope, freq. instantânea) em bins log,
    em dB relativo à mediana -> linhas de baud/keying aparecem como picos."""
    sig = sig - sig.mean()
    w = np.hanning(len(sig))
    S = np.abs(rfft(sig * w))
    f = rfftfreq(len(sig), 1 / fs)
    band = (f >= fmin) & (f <= fmax)
    med = np.median(S[band]) + EPS
    pooled = 20 * np.log10(_log_pool_max(S, f, fmin, fmax, nb) / med + EPS)
    sb = S[band]
    k = int(np.argmax(sb))
    top_f = np.log10(f[band][k])
    top_p = 20 * np.log10(sb[k] / med)
    return np.clip(pooled, -20, 60), np.array([top_f, min(top_p, 80)])


def _kurt(v):
    v = v - v.mean()
    s2 = np.mean(v ** 2) + EPS
    return np.mean(v ** 4) / s2 ** 2


def _skew(v):
    v = v - v.mean()
    return np.mean(v ** 3) / (np.mean(v ** 2) + EPS) ** 1.5


# ---------------------------------------------------------------- features
def extract(x, fs=FS):
    """Retorna vetor de features (float32, tamanho fixo) para um trecho de áudio."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if len(x) < fs:  # mínimo 1 s
        rng = np.random.default_rng(0)
        x = np.concatenate([x, rng.normal(0, 1e-3 * (x.std() + 1e-9), fs - len(x))])
    x = x / (np.sqrt(np.mean(x ** 2)) + EPS)
    out = []

    # ---------- STFT
    _, _, Z = sps.stft(x, fs, window="hann", nperseg=NFFT, noverlap=NFFT - HOP,
                       boundary=None, padded=False)
    P = np.abs(Z) ** 2 + EPS                     # (513, T)
    T = P.shape[1]
    f = _F

    # mel / MFCC
    M = _FB @ P
    logM = 10 * np.log10(M + EPS)
    out += [logM.mean(1) - logM.mean(1).max(), logM.std(1)]
    mfcc = dct(logM, type=2, axis=0, norm="ortho")[:20]
    out += [mfcc.mean(1), mfcc.std(1)]
    dm = np.diff(mfcc, axis=1)
    out.append(dm.std(1) if dm.shape[1] else np.zeros(20))

    # espectro médio
    S = P.mean(1)
    Sdb = 10 * np.log10(S)
    Sdb -= Sdb.max()
    occ = []
    for th in (-6, -10, -20, -30):
        m = Sdb > th
        ix = np.nonzero(m)[0]
        occ += [m.mean(), (ix[-1] - ix[0]) * _DF / 1000 if len(ix) else 0.0]
    out.append(np.array(occ))
    p = S / S.sum()
    cen = (f * p).sum()
    spread = np.sqrt(((f - cen) ** 2 * p).sum())
    cums = np.cumsum(p)
    rolls = [f[min(np.searchsorted(cums, q), len(f) - 1)] for q in (0.05, 0.15, 0.5, 0.85, 0.95)]
    flat = np.exp(np.mean(np.log(S))) / np.mean(S)
    out.append(np.array([cen / 1000, spread / 1000, _skew_w(f, p), np.log10(_kurt_w(f, p) + EPS),
                         *(r / 1000 for r in rolls), flat, np.log10(flat + EPS)]))

    # banda principal (região contígua em torno do pico acima de -25 dB, suavizado)
    sm = np.convolve(Sdb, np.ones(5) / 5, mode="same")
    k0 = int(np.argmax(sm))
    lo = k0
    while lo > 0 and sm[lo - 1] > -25:
        lo -= 1
    hi = k0
    while hi < len(sm) - 1 and sm[hi + 1] > -25:
        hi += 1
    f_lo, f_hi = f[lo], f[hi]
    if f_hi - f_lo < 300:
        c = 0.5 * (f_lo + f_hi)
        f_lo, f_hi = max(c - 150, 0), min(c + 150, fs / 2)
    out.append(np.array([f_lo / 1000, f_hi / 1000, (f_hi - f_lo) / 1000, f[k0] / 1000]))

    # estrutura de tons: picos do espectro médio
    det = Sdb - sps.medfilt(Sdb, 31)
    pk, pr = sps.find_peaks(Sdb, prominence=3, distance=2)
    prom = pr["prominences"] if len(pk) else np.array([])
    n6, n12, n20 = (prom > 6).sum(), (prom > 12).sum(), (prom > 20).sum()
    top = pk[np.argsort(prom)[::-1][:4]] if len(pk) else np.array([], int)
    topf = np.sort(f[top]) if len(top) else np.array([])
    tf = np.zeros(4)
    tf[:len(topf)] = topf / 1000
    sp = np.zeros(3)
    if len(topf) > 1:
        d = np.diff(topf)
        sp[:len(d)] = d / 1000
    out.append(np.array([n6, n12, n20, np.log1p(n6), prom.max() if len(prom) else 0,
                         *tf, *sp]))

    # autocorrelação do espectro destendenciado -> espaçamento entre tons
    s = det - det.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:len(s) - 1 + 129]
    ac = ac / (ac[0] + EPS)
    out.append(ac[1:129].reshape(32, 4).mean(1))

    # por quadro
    fr_flat = np.exp(np.mean(np.log(P), 0)) / np.mean(P, 0)
    fr_cen = (f[:, None] * P).sum(0) / P.sum(0)
    Pn = P / P.sum(0, keepdims=True)
    flux = np.sqrt((np.diff(Pn, axis=1) ** 2).sum(0)) if T > 1 else np.zeros(1)
    band_ix = (f > 80) & (f < fs / 2 - 80)
    pkf = f[band_ix][np.argmax(P[band_ix], 0)]
    rel = pkf - np.median(pkf)
    h, _ = np.histogram(rel, bins=np.linspace(-1500, 1500, 41))
    h = h / max(T, 1)
    habs, _ = np.histogram(pkf, bins=np.arange(0, fs / 2 + 32, 32))
    habs = habs / max(T, 1)
    ent = -(habs[habs > 0] * np.log(habs[habs > 0])).sum()
    dpk = np.diff(pkf)
    nz = dpk[np.abs(dpk) > 0]
    out.append(np.array([fr_flat.mean(), fr_flat.std(), fr_cen.std() / 1000, flux.mean(), flux.std(),
                         ent, (habs > 0.05).sum(), (habs > 0.15).sum(),
                         np.mean(np.abs(dpk) <= 2 * _DF) if len(dpk) else 1,
                         np.abs(np.mean(np.sign(nz))) if len(nz) else 0,
                         np.median(np.abs(dpk)) / 1000 if len(dpk) else 0,
                         1 - h.sum()]))
    out.append(h)

    # estrutura temporal lenta (energia por quadro, 62.5 Hz)
    e = 10 * np.log10(P.sum(0))
    lo_e, hi_e = np.percentile(e, 10), np.percentile(e, 90)
    on = e > (lo_e + hi_e) / 2
    dur = len(x) / fs
    trans = np.count_nonzero(np.diff(on.astype(int))) / dur
    el = 10 ** ((e - e.max()) / 10)
    el = el - el.mean()
    acf = np.correlate(el, el, "full")[len(el) - 1:]
    acf = acf / (acf[0] + EPS)
    lags = np.unique(np.geomspace(1, max(2, T // 2), 16).astype(int))
    a16 = np.zeros(16)
    a16[:len(lags)] = acf[np.minimum(lags, len(acf) - 1)]
    out.append(np.array([hi_e - lo_e, np.percentile(e, 99) - np.percentile(e, 1), on.mean(),
                         np.log1p(trans), _kurt(el), *a16]))
    fr_rate = fs / HOP
    ms, _ = _line_spectrum_feats(e - e.mean(), fr_rate, 0.3, fr_rate / 2 - 0.5, 16)
    out.append(ms)

    # sinal analítico limitado à banda principal
    X = fft(x)
    fx = np.fft.fftfreq(len(x), 1 / fs)
    mask = (fx >= f_lo) & (fx <= f_hi)
    a = ifft(np.where(mask, 2 * X, 0))
    env = np.abs(a) + EPS
    out.append(np.array([env.std() / env.mean(), np.log10(_kurt(env)),
                         min(10 * np.log10(np.max(env ** 2) / np.mean(env ** 2)), 30)]))
    l1, t1 = _line_spectrum_feats(env ** 2, fs, 2, 4000, 40)
    out += [l1, t1]

    # frequência instantânea
    ph = np.angle(a[1:] * np.conj(a[:-1])) * fs / (2 * np.pi)
    ifr = np.convolve(ph, np.ones(8) / 8, mode="same")
    rel_if = ifr - np.median(ifr)
    hif, _ = np.histogram(rel_if, bins=np.linspace(-1500, 1500, 31))
    hif = hif / len(rel_if)
    out.append(hif)
    out.append(np.array([np.std(rel_if) / 1000, _kurt(rel_if) if np.std(rel_if) > 0 else 0]))
    l2, t2 = _line_spectrum_feats(ifr, fs, 2, 4000, 40)
    out += [l2, t2]

    # potências do sinal analítico: linhas em a^n -> PSK de ordem n / portadora
    an = a / (np.sqrt(np.mean(np.abs(a) ** 2)) + EPS)
    pw = []
    for n in (1, 2, 4, 8):
        Sp = np.abs(fft(an ** n))
        pw.append(min(20 * np.log10(Sp.max() / (np.median(Sp) + EPS)), 80))
    out.append(np.array(pw))

    # zcr
    zcr = np.mean(np.abs(np.diff(np.signbit(x).astype(int))))
    out.append(np.array([zcr]))

    v = np.concatenate([np.ravel(o) for o in out]).astype(np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=80.0, neginf=-80.0)


def _skew_w(f, p):
    m = (f * p).sum()
    s = np.sqrt(((f - m) ** 2 * p).sum()) + EPS
    return (((f - m) / s) ** 3 * p).sum()


def _kurt_w(f, p):
    m = (f * p).sum()
    s = np.sqrt(((f - m) ** 2 * p).sum()) + EPS
    return (((f - m) / s) ** 4 * p).sum()


N_FEATURES = len(extract(np.random.default_rng(1).normal(size=FS * 2)))


# ---------------------------------------------------------------- aumentação
def augment(x, rng, fs=FS, shift=True):
    """Desloca em frequência (sintonia SSB diferente) e adiciona ruído."""
    y = x.astype(np.float64)
    if shift and rng.random() < 0.7:
        f0 = rng.uniform(-250, 250)
        t = np.arange(len(y)) / fs
        y = np.real(sps.hilbert(y) * np.exp(2j * np.pi * f0 * t))
    if rng.random() < 0.6:
        snr = rng.uniform(3, 25)
        p = np.mean(y ** 2)
        n = rng.normal(0, 1, len(y))
        if rng.random() < 0.5:  # ruído de receptor SSB (~300-2700 Hz)
            b, a_ = sps.butter(4, [250, 2900], btype="band", fs=fs)
            n = sps.lfilter(b, a_, n)
        y = y + n * np.sqrt(p / (np.mean(n ** 2) + EPS) / 10 ** (snr / 10))
    return y.astype(np.float32)


def synthetic_noise(rng, n_items=150, fs=FS, win_s=WIN_S):
    """Janelas de ruído/estática para a classe 'sem sinal'."""
    N = int(win_s * fs)
    items = []
    for i in range(n_items):
        kind = i % 5
        w = rng.normal(0, 1, N)
        if kind == 1:  # ruído rosa
            W = np.fft.rfft(w)
            W /= np.sqrt(np.arange(1, len(W) + 1))
            w = np.fft.irfft(W, N)
        elif kind == 2:  # receptor SSB
            lo, hi = rng.uniform(100, 400), rng.uniform(2300, 3200)
            b, a_ = sps.butter(4, [lo, hi], btype="band", fs=fs)
            w = sps.lfilter(b, a_, w)
        elif kind == 3:  # NFM com squelch aberto (de-ênfase)
            b, a_ = sps.butter(1, rng.uniform(1500, 4000), fs=fs)
            w = sps.lfilter(b, a_, w)
        elif kind == 4:  # estática impulsiva (QRN)
            imp = (rng.random(N) < rng.uniform(0.001, 0.01)) * rng.normal(0, 20, N)
            b, a_ = sps.butter(4, [200, 3000], btype="band", fs=fs)
            w = sps.lfilter(b, a_, w + imp)
        items.append(w.astype(np.float32))
    return items


# ---------------------------------------------------------------- simulação de microfone (app Android)
def _band_noise(rng, n, lo, hi, fs=FS, order=2):
    b, a = sps.butter(order, [lo, hi], btype="band", fs=fs)
    return sps.lfilter(b, a, rng.normal(0, 1, n))


def _ambient(rng, n, fs=FS):
    """Ruído de campo: vento com rajadas, falatório, ruído rosa ou trânsito."""
    kind = rng.integers(0, 4)
    t = np.arange(n) / fs
    if kind == 0:  # vento: grave, com rajadas lentas
        b, a = sps.butter(2, rng.uniform(80, 400), fs=fs)
        w = sps.lfilter(b, a, rng.normal(0, 1, n))
        gust = 0.4 + np.abs(np.convolve(rng.normal(0, 1, n // 800 + 2), np.ones(3) / 3, "same"))
        w *= np.interp(np.linspace(0, len(gust) - 1, n), np.arange(len(gust)), gust)
    elif kind == 1:  # falatório distante: faixa de voz modulada no ritmo silábico
        w = np.zeros(n)
        for _ in range(rng.integers(2, 6)):
            env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2.5, 6) * t + rng.uniform(0, 6.3))
            w += _band_noise(rng, n, rng.uniform(200, 500), rng.uniform(1800, 3500)) * env ** 2
    elif kind == 2:  # ruído rosa
        W = np.fft.rfft(rng.normal(0, 1, n))
        W /= np.sqrt(np.arange(1, len(W) + 1))
        w = np.fft.irfft(W, n)
    else:  # trânsito/motor: grave + harmônicos de rotação
        f0 = rng.uniform(25, 90)
        w = _band_noise(rng, n, 30, 900) + sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.3)) / k
                                              for k in range(1, 6)) * 0.3
    return w


def augment_mic(x, rng, fs=FS, shift=False):
    """Simula: alto-falante de rádio portátil -> ar/sala -> microfone de celular, em campo."""
    y = x.astype(np.float64)
    n = len(y)
    if shift and rng.random() < 0.5:
        t = np.arange(n) / fs
        y = np.real(sps.hilbert(y) * np.exp(2j * np.pi * rng.uniform(-200, 200) * t))
    # alto-falante pequeno: passa-faixa + ressonância do gabinete
    lo, hi = rng.uniform(150, 550), rng.uniform(2800, 6500)
    b, a = sps.butter(2, [lo, hi], btype="band", fs=fs)
    y = sps.lfilter(b, a, y)
    bp, ap = sps.iirpeak(rng.uniform(500, 3000), rng.uniform(1.0, 4.0), fs=fs)
    y = y + (10 ** (rng.uniform(2, 10) / 20) - 1) * sps.lfilter(bp, ap, y)
    if rng.random() < 0.5:  # volume alto: distorção do alto-falante
        d = rng.uniform(0.8, 3.5)
        y = np.tanh(d * y / (2 * np.std(y) + EPS))
    # sala / ar livre: resposta impulsiva sintética
    if rng.random() < 0.85:
        rt60 = rng.uniform(0.08, 0.6)
        m = max(int(rt60 * fs), 32)
        tt = np.arange(m) / fs
        rir = rng.normal(0, 1, m) * np.exp(-6.9 * tt / rt60) * rng.uniform(0.05, 0.5)
        rir[0] = 1.0
        y = sps.fftconvolve(y, rir)[:n]
    # ruído de campo
    if rng.random() < 0.9:
        amb = _ambient(rng, n, fs)
        snr = rng.uniform(3, 30)
        y = y + amb * np.sqrt(np.mean(y ** 2) / (np.mean(amb ** 2) + EPS) / 10 ** (snr / 10))
    # microfone do celular: passa-altas, corte alto suave, ganho variando (AGC)
    b, a = sps.butter(2, rng.uniform(70, 200), btype="high", fs=fs)
    y = sps.lfilter(b, a, y)
    b, a = sps.butter(2, rng.uniform(5500, 7600), fs=fs)
    y = sps.lfilter(b, a, y)
    if rng.random() < 0.4:
        g = 1 + rng.uniform(0.1, 0.5) * np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * np.arange(n) / fs)
        y = y * g
    y = y + rng.normal(0, 1, n) * np.std(y) * 10 ** (-rng.uniform(35, 55) / 20)  # ruído próprio do mic
    return y.astype(np.float32)
