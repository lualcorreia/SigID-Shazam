"""
Exporta o modelo mobile + metadados + listas QRG para o app Android (SigID-Android).

  python export_android.py [models/sigid_model_mobile.joblib] [caminho/do/projeto/android]

Formato binário (little-endian): int32 n_arrays; para cada: int32 len_nome, nome utf-8,
int32 dtype (0=f32, 1=f16, 2=i32, 3=i16, 4=f64), int32 ndim, int32 dims..., dados.
"""
import json, os, struct, sys, gzip
import numpy as np
import joblib

import sigfeatures as sf_
from sigid_core import parse_freqs, mode_family, model_proba, _temper, W_MLP
import qrg_hints as QH

ROOT = os.path.dirname(os.path.abspath(__file__))
DT = {np.dtype("float32"): 0, np.dtype("float16"): 1, np.dtype("int32"): 2, np.dtype("int16"): 3,
      np.dtype("float64"): 4}


def write_bin(path, arrays):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<i", len(arrays)))
        for name, a in arrays.items():
            a = np.ascontiguousarray(a)
            nb = name.encode()
            fh.write(struct.pack("<i", len(nb)) + nb)
            fh.write(struct.pack("<ii", DT[a.dtype], a.ndim))
            fh.write(struct.pack("<" + "i" * a.ndim, *a.shape))
            fh.write(a.astype(a.dtype.newbyteorder("<")).tobytes())
    print(f"  {os.path.basename(path)}: {os.path.getsize(path) / 1e6:.1f} MB")


def lags_for(T):
    return np.unique(np.geomspace(1, max(2, T // 2), 16).astype(int))


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "sigid_model_mobile.joblib")
    default = os.path.join(os.path.dirname(ROOT), "android")  # layout do repositório (pc/ e android/)
    if not os.path.isdir(default):
        default = os.path.join(os.path.dirname(ROOT), "SigID-Android")
    proj = sys.argv[2] if len(sys.argv) > 2 else default
    assets = os.path.join(proj, "app", "src", "main", "assets")
    tests = os.path.join(proj, "app", "src", "test", "resources")
    os.makedirs(assets, exist_ok=True)
    os.makedirs(tests, exist_ok=True)
    m = joblib.load(mp)
    classes = list(m["classes"])
    print(f"Modelo {mp}: {len(classes)} classes, {len(m['mlps'])} MLPs, mobile={m.get('mobile')}")

    A = {}
    A["fb"] = sf_._FB.astype(np.float64)
    T = (int(sf_.WIN_S * sf_.FS) - sf_.NFFT) // sf_.HOP + 1
    A["lags"] = lags_for(T).astype(np.int32)
    A["scaler_mean"] = m["scaler"].mean_.astype(np.float64)
    A["scaler_scale"] = m["scaler"].scale_.astype(np.float64)
    for k, mlp in enumerate(m["mlps"]):
        for l, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            A[f"mlp{k}_W{l}"] = W.astype(np.float32)
            A[f"mlp{k}_b{l}"] = b.astype(np.float32)
    mod = m["mod"]
    for l, (W, b) in enumerate(zip(mod.coefs_, mod.intercepts_)):
        A[f"mod_W{l}"] = W.astype(np.float32)
        A[f"mod_b{l}"] = b.astype(np.float32)
    pca = m["pca"]
    A["pca_mean"] = pca.mean_.astype(np.float32)
    A["pca_comp"] = pca.components_.astype(np.float32)
    knn = m["knn"]
    Xk = knn._fit_X.astype(np.float32)
    A["knn_X"] = Xk.astype(np.float16)
    A["knn_y"] = knn._y.astype(np.int16)
    write_bin(os.path.join(assets, "model.bin"), A)

    meta = {"classes": classes, "mod_classes": list(mod.classes_), "n_mlp": len(m["mlps"]),
            "n_layers": len(m["mlps"][0].coefs_), "n_mod_layers": len(mod.coefs_),
            "T": m.get("T", 2.0), "w_mlp": W_MLP, "knn_k": knn.n_neighbors,
            "fs": sf_.FS, "win_s": sf_.WIN_S, "feature_version": sf_.FEATURE_VERSION,
            "report": {k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "dtype") else str(v))
                       for k, v in (m.get("report") or {}).items()},
            "trained": m.get("trained"), "mobile": bool(m.get("mobile")), "signals": []}
    for c in classes:
        info = m["meta"].get(c, {})
        f = info.get("fields", {})
        meta["signals"].append({
            "fields": {k: f.get(k, "") for k in ("frequencies", "mode", "modulation", "bandwidth", "location")
                       if f.get(k)},
            "description": info.get("description", ""), "url": info.get("url", ""),
            "categories": info.get("categories", []),
            "ranges": parse_freqs(f.get("frequencies", "")), "fams": sorted(mode_family(f.get("mode", "")))})
    with open(os.path.join(assets, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    print(f"  meta.json: {os.path.getsize(os.path.join(assets, 'meta.json')) / 1e6:.1f} MB")

    # listas QRG com as classes já resolvidas (perfil de correspondência deduplicado)
    q = QH.QRGHints(classes, m["meta"])
    prof_id, profs = {}, []
    lines = []
    for e in q.entries:
        cls, fam = q.match(e)
        key = (tuple(sorted((i, round(s, 3)) for i, s in cls.items())), tuple(sorted(fam.items())))
        if key not in prof_id:
            prof_id[key] = len(profs)
            profs.append(key)
        clean = lambda s: (s or "").replace("\t", " ").replace("\n", " ")
        lines.append("\t".join([f"{e['f']:.2f}", clean(e["station"]), clean(e["call"]), clean(QH.describe(e))[:90],
                                clean(e["emis"] or e["mode"]), "" if e["t0"] is None else str(e["t0"]),
                                "" if e["t1"] is None else str(e["t1"]), e["days"], e["src"], clean(e["loc"]),
                                str(prof_id[key])]))
    with gzip.open(os.path.join(assets, "qrg.tsv.gz"), "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(assets, "qrg_profiles.tsv"), "w", encoding="utf-8") as fh:
        for cl, fa in profs:
            fh.write(",".join(f"{i}:{s}" for i, s in cl) + "\t" + ",".join(f"{k}:{v}" for k, v in fa) + "\n")
    print(f"  qrg.tsv.gz: {len(lines)} entradas, {len(profs)} perfis, "
          f"{os.path.getsize(os.path.join(assets, 'qrg.tsv.gz')) / 1e6:.1f} MB")
    fam_cls = [QH._family_of(m["meta"].get(c, {}).get("fields", {}).get("modulation", "")) or "" for c in classes]
    with open(os.path.join(assets, "class_fams.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(fam_cls))

    # vetores de teste: áudio, features e probabilidades esperadas
    rng = np.random.default_rng(5)
    import glob
    N = int(sf_.WIN_S * sf_.FS)
    xs = []
    for d in ("Radio Teletype (RTTY)", "Morse Code (CW)", "STANAG 4285", "Single Sideband Voice",
              "GMDSS Digital Selective Calling", "Digital Radio Mondiale (DRM)"):
        fs_ = sorted(glob.glob(os.path.join(ROOT, "data", "audio", d, "*")))
        if fs_:
            x = sf_.load_audio(fs_[0])
            x = np.tile(x, int(np.ceil(N / len(x))))[:N] if len(x) < N else x[:N]
            xs.append(sf_.augment_mic(x, rng))
    xs.append(rng.normal(0, 1, N).astype(np.float32))
    t = np.arange(N) / sf_.FS
    xs.append((np.sin(2 * np.pi * 1000 * t) * 0.3).astype(np.float32))
    X = np.stack(xs).astype(np.float32)
    F = np.stack([sf_.extract(x) for x in X])
    P = model_proba(m, F)
    PM = m["mod"].predict_proba(np.clip(m["scaler"].transform(F), -8, 8))
    write_bin(os.path.join(tests, "vectors.bin"), {"x": X, "feat": F.astype(np.float32),
                                                   "p": P.astype(np.float32), "pm": PM.astype(np.float32)})
    print("  vetores de teste:", len(X))


if __name__ == "__main__":
    main()
