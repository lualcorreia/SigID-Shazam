"""Compara modelo PC x modelo mobile em áudio passado pelo microfone simulado."""
import glob
import numpy as np
import sigfeatures as s
from sigid_core import Identifier

pc = Identifier("models/sigid_model.joblib")
mo = Identifier("models/sigid_model_mobile.joblib")
rng = np.random.default_rng(99)
sig = ["Radio Teletype (RTTY)", "Morse Code (CW)", "STANAG 4285", "GMDSS Digital Selective Calling",
       "Slow-Scan Television (SSTV)", "Digital Mobile Radio (DMR)", "POCSAG", "Single Sideband Voice",
       "WEFAX", "High Frequency Data Link (HFDL)"]
res = {"PC": [0, 0, 0], "mobile": [0, 0, 0]}
n = 0
for d in sig:
    files = sorted(glob.glob(f"data/audio/{d}/*"))
    if not files:
        continue
    x = s.load_audio(files[-1])[:64000]
    for _ in range(3):
        y = s.augment_mic(x, rng)
        n += 1
        for name, I in (("PC", pc), ("mobile", mo)):
            p, pm = I.proba(y)
            top = [l for l, _ in I.top(p, 5)]
            res[name][0] += top[0] == d
            res[name][1] += d in top
print(f"{n} janelas via microfone simulado ({len(sig)} sinais x 3):")
for k, (t1, t5, _) in res.items():
    print(f"  modelo {k:7s}: top-1 {t1}/{n}   top-5 {t5}/{n}")
