"""Enxuga os modelos salvos (remove estado de treino) e confere que as previsões não mudam."""
import os, shutil
import numpy as np
import joblib
from sigid_core import slim, model_proba

ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
for name in ("sigid_model.joblib", "sigid_model_mobile.joblib"):
    p = os.path.join(ROOT, "models", name)
    if not os.path.exists(p):
        continue
    before = os.path.getsize(p)
    m = joblib.load(p)
    X = rng.normal(0, 1, (64, m["n_features"])).astype(np.float32) * m["scaler"].scale_ + m["scaler"].mean_
    p0 = model_proba(m, X)
    pm0 = m["mod"].predict_proba(np.clip(m["scaler"].transform(X), -8, 8))
    shutil.copy2(p, p + ".bak")
    joblib.dump(slim(m), p, compress=3)
    m2 = joblib.load(p)
    d = np.abs(model_proba(m2, X) - p0).max()
    dm = np.abs(m2["mod"].predict_proba(np.clip(m2["scaler"].transform(X), -8, 8)) - pm0).max()
    assert d == 0 and dm == 0, (d, dm)
    os.remove(p + ".bak")
    print(f"{name}: {before / 1e6:.1f} MB -> {os.path.getsize(p) / 1e6:.1f} MB (previsões idênticas)")
