"""Carrega o modelo treinado e identifica trechos de áudio."""
import os
import numpy as np
import joblib
import sigfeatures as sf_

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "sigid_model.joblib")
W_MLP = 0.65  # peso da rede neural no ensemble (o resto é kNN)


def fit_model(X, y, n_mlp=1, seed=0):
    """Ensemble: n redes MLP + kNN (cosseno) num espaço PCA. Leve em memória mesmo com 500+ classes."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    sc = StandardScaler().fit(X)
    Z = np.clip(sc.transform(X), -8, 8).astype(np.float32)
    mlps = []
    for k in range(n_mlp):
        m = MLPClassifier(hidden_layer_sizes=(768, 384), alpha=2e-3, batch_size=256,
                          learning_rate_init=1e-3, max_iter=120, early_stopping=True,
                          validation_fraction=0.08, n_iter_no_change=10, random_state=seed + k)
        try:
            m.fit(Z, y)
        except ValueError:  # alguma classe pequena demais p/ separar validação estratificada
            m.set_params(early_stopping=False, max_iter=45)
            m.fit(Z, y)
        mlps.append(m)
        print(f"    MLP {k + 1}/{n_mlp}: {m.n_iter_} épocas", flush=True)
    pca = PCA(n_components=min(96, Z.shape[1]), random_state=seed).fit(Z)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine", n_jobs=-1)
    knn.fit(pca.transform(Z).astype(np.float32), y)
    classes = mlps[0].classes_
    assert all((m.classes_ == classes).all() for m in mlps) and (knn.classes_ == classes).all()
    return {"scaler": sc, "mlps": mlps, "pca": pca, "knn": knn, "classes": list(classes)}


_TRAIN_ONLY = ("_optimizer", "_best_coefs", "_best_intercepts", "loss_curve_", "validation_scores_",
               "_no_improvement_count", "best_loss_", "best_validation_score_")


def slim(model):
    """Remove do modelo o que só serve para treinar (estado do Adam, cópias do early stopping)."""
    for m in list(model.get("mlps", [])) + ([model["mod"]] if "mod" in model else []):
        for a in _TRAIN_ONLY:
            if hasattr(m, a):
                try:
                    delattr(m, a)
                except AttributeError:
                    pass
    return model


def _temper(p, T):
    """Escala de temperatura: redes neurais costumam ser confiantes demais."""
    q = np.power(np.clip(p, 1e-12, 1), 1.0 / T)
    return q / q.sum(axis=-1, keepdims=True)


def model_proba(model, X):
    Z = np.clip(model["scaler"].transform(np.atleast_2d(X)), -8, 8).astype(np.float32)
    pm = np.mean([_temper(m.predict_proba(Z), model.get("T", 2.0)) for m in model["mlps"]], axis=0)
    pk = model["knn"].predict_proba(model["pca"].transform(Z).astype(np.float32))
    return W_MLP * pm + (1 - W_MLP) * pk


class Identifier:
    def __init__(self, path=MODEL_PATH):
        m = joblib.load(path)
        if m.get("feature_version") != sf_.FEATURE_VERSION or "mlps" not in m:
            raise RuntimeError("Modelo antigo/incompatível; rode train.py de novo.")
        self.m = m
        self.scaler, self.mod = m["scaler"], m["mod"]
        self.classes = np.array(m["classes"])
        self.meta = m["meta"]
        self.report = m.get("report", {})
        self.fs = m["fs"]
        self.win_s = m["win_s"]

    def proba(self, x16):
        """x16: áudio mono já em FS (16 kHz). Retorna (p_sinais, p_modulação)."""
        F = sf_.extract(x16)[None]
        p = model_proba(self.m, F)[0]
        Z = np.clip(self.scaler.transform(F), -8, 8)
        pm = self.mod.predict_proba(Z)[0]
        return p, pm

    def top(self, p, k=5):
        o = np.argsort(-p)[:k]
        return [(self.classes[i], float(p[i])) for i in o]

    def top_mod(self, pm, k=3):
        o = np.argsort(-pm)[:k]
        return [(self.mod.classes_[i], float(pm[i])) for i in o]

    def identify_array(self, x, sr):
        x16 = sf_.to_mono_fs(x, sr)
        segs = sf_.segments(x16, hop_s=2.0, max_n=60, keep_db=20)
        P, PM = [], []
        for s in segs:
            p, pm = self.proba(s)
            P.append(p); PM.append(pm)
        P, PM = np.array(P), np.array(PM)
        return P.mean(0), PM.mean(0), P

    def identify_file(self, path):
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return self.identify_array(x, sr)

    def info(self, label):
        return self.meta.get(label, {})

# ---------------------------------------------------------------- dicas (priors)
import re as _re

_UNIT = {"ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0}
_FREQ_RE = _re.compile(r"(\d+(?:[.,]\d+)?)\s*(ghz|mhz|khz|hz)", _re.I)


def parse_freqs(text):
    """'3 MHz,30 MHz' -> faixa (3e6, 30e6); '8.992 MHz' -> [(8.75e6, 9.2e6)]."""
    vals = [float(v.replace(",", ".")) * _UNIT[u.lower()] for v, u in _FREQ_RE.findall(text or "")]
    if not vals:
        return []
    if len(vals) == 2 and vals[1] > vals[0]:
        return [(vals[0], vals[1])]
    return [(v * 0.97 - 5e3, v * 1.03 + 5e3) for v in vals]


MODE_FAMILY = {"USB": "SSB", "LSB": "SSB", "SSB": "SSB", "CW": "SSB",
               "NFM": "FM", "FM": "FM", "WFM": "FM", "AM": "AM"}


def mode_family(text):
    t = (text or "").upper()
    fams = {fam for k, fam in MODE_FAMILY.items() if _re.search(rf"\b{k}\b", t)}
    return fams


class Priors:
    """Pesos por classe a partir da frequência sintonizada e do modo de demodulação."""

    def __init__(self, ident):
        self.ident = ident
        self.ranges = [parse_freqs(ident.info(c).get("fields", {}).get("frequencies", ""))
                       for c in ident.classes]
        self.fams = [mode_family(ident.info(c).get("fields", {}).get("mode", ""))
                     for c in ident.classes]

    def weights(self, freq_hz=None, mode=None):
        w = np.ones(len(self.ident.classes))
        if freq_hz:
            for i, rr in enumerate(self.ranges):
                if rr and not any(a <= freq_hz <= b for a, b in rr):
                    w[i] *= 0.2
        fam = MODE_FAMILY.get((mode or "").upper())
        if fam:
            for i, fs in enumerate(self.fams):
                if fs and fam not in fs:
                    w[i] *= 0.4
        return w

    @staticmethod
    def apply(p, w):
        q = p * w
        return q / (q.sum() + 1e-12)
