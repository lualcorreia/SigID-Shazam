"""
Treina o identificador de sinais a partir de data/manifest.json + data/audio.

  python train.py            # extrai features (com cache), avalia e salva models/sigid_model.joblib
  python train.py --no-eval  # pula a validação cruzada
"""
import argparse, hashlib, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import joblib

import sigfeatures as sf_

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "features")
MODELS = os.path.join(ROOT, "models")
NOISE_LABEL = "Ruído / sem sinal"
N_AUG = 1  # cópias aumentadas por janela


def _cache_path(path):
    h = hashlib.md5(f"{path}|{sf_.FEATURE_VERSION}|{N_AUG}".encode()).hexdigest()[:16]
    return os.path.join(CACHE, h + ".npz")


def file_features(args):
    path, ssb_like = args
    cp = _cache_path(path)
    if os.path.exists(cp):
        d = np.load(cp)
        return path, d["X"], d["aug"]
    try:
        x = sf_.load_audio(path)
    except Exception as e:
        return path, None, f"erro ao ler: {e}"
    if len(x) < sf_.FS * 0.5 or np.std(x) < 1e-6:
        return path, None, "curto/silencioso"
    rng = np.random.default_rng(int(hashlib.md5(path.encode()).hexdigest()[:8], 16))
    X, aug = [], []
    for seg in sf_.segments(x):
        X.append(sf_.extract(seg)); aug.append(0)
        for _ in range(N_AUG):
            X.append(sf_.extract(sf_.augment(seg, rng, shift=ssb_like))); aug.append(1)
    X = np.vstack(X).astype(np.float32)
    aug = np.array(aug, np.int8)
    np.savez_compressed(cp, X=X, aug=aug)
    return path, X, aug


N_MIC = 2  # cópias "alto-falante -> sala -> microfone" por janela (modelo mobile)


def file_features_mic(args):
    path, ssb_like = args
    h = hashlib.md5(f"{path}|{sf_.FEATURE_VERSION}|mic{N_MIC}".encode()).hexdigest()[:16]
    cp = os.path.join(CACHE, h + ".npz")
    if os.path.exists(cp):
        d = np.load(cp)
        return path, d["X"], d["aug"]
    try:
        x = sf_.load_audio(path)
    except Exception as e:
        return path, None, f"erro ao ler: {e}"
    if len(x) < sf_.FS * 0.5 or np.std(x) < 1e-6:
        return path, None, "curto/silencioso"
    rng = np.random.default_rng(int(hashlib.md5((path + "mic").encode()).hexdigest()[:8], 16))
    X = [sf_.extract(sf_.augment_mic(seg, rng, shift=ssb_like))
         for seg in sf_.segments(x) for _ in range(N_MIC)]
    X = np.vstack(X).astype(np.float32)
    aug = np.full(len(X), 2, np.int8)
    np.savez_compressed(cp, X=X, aug=aug)
    return path, X, aug


def modulation_label(fields):
    m = (fields.get("modulation") or "").upper()
    m = re.split(r"[,/;]| OR ", m)[0].strip()
    m = re.sub(r"\(.*?\)", "", m).strip()
    if not re.search(r"[A-Z]", m) or re.search(r"[|=]", m) or m.startswith(("UNKNOWN", "N/A", "VARIOUS", "?")):
        return None
    aliases = {"MFSK": "MFSK", "8-FSK": "MFSK", "8FSK": "MFSK", "IFK": "MFSK",
               "USB": "AM", "LSB": "AM", "J3E": "AM", "ANALOG OFDM": "OFDM",
               "GFSK": "FSK", "2FSK": "FSK", "4FSK": "4FSK", "C4FM": "4FSK",
               "BPSK": "PSK", "QPSK": "PSK", "8PSK": "PSK", "DBPSK": "PSK", "DQPSK": "PSK",
               "PSK": "PSK", "OOK": "OOK/ASK", "ASK": "OOK/ASK", "CW": "OOK/ASK",
               "NFM": "FM", "WFM": "FM", "FM": "FM", "AM": "AM", "SSB": "AM",
               "OFDM": "OFDM", "COFDM": "OFDM", "QAM": "QAM", "FSK": "FSK", "AFSK": "AFSK",
               "MSK": "MSK", "GMSK": "MSK", "PULSE": "PULSOS", "CHIRP": "CHIRP"}
    for k, v in aliases.items():
        if m == k or m.startswith(k):
            return v
    return m[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--mod-only", action="store_true", help="retreina só o modelo de modulação")
    ap.add_argument("--mobile", action="store_true",
                    help="inclui simulação de microfone (app Android) -> models/sigid_model_mobile.joblib")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    global MP
    MP = os.path.join(MODELS, "sigid_model_mobile.joblib" if args.mobile else "sigid_model.joblib")
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    man = json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))
    signals = man["signals"]
    jobs, labels = [], {}
    for f in man["files"]:
        p = os.path.join(DATA, f["path"])
        if not os.path.exists(p):
            continue
        fields = signals.get(f["signal"], {}).get("fields", {})
        mode = (fields.get("mode") or "").upper()
        ssb_like = any(k in mode for k in ("USB", "LSB", "SSB", "CW"))
        jobs.append((p, ssb_like))
        labels[p] = f["signal"]
    print(f"{len(jobs)} arquivos disponíveis, {len(set(labels.values()))} sinais")

    t0 = time.time()
    Xs, ys, gs, augs = [], [], [], []
    bad = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (p, X, aug) in enumerate(ex.map(file_features, jobs, chunksize=4), 1):
            if X is None:
                bad += 1
            else:
                Xs.append(X); ys += [labels[p]] * len(X); gs += [p] * len(X); augs.append(aug)
            if i % 100 == 0 or i == len(jobs):
                print(f"  features {i}/{len(jobs)}  ({time.time() - t0:.0f}s, {bad} ignorados)", flush=True)

    if args.mobile:
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            for i, (p, X, aug) in enumerate(ex.map(file_features_mic, jobs, chunksize=4), 1):
                if X is not None:
                    Xs.append(X); ys += [labels[p]] * len(X); gs += [p] * len(X); augs.append(aug)
                if i % 100 == 0 or i == len(jobs):
                    print(f"  features microfone {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(42)
    noise = sf_.synthetic_noise(rng)
    if args.mobile:  # no celular, "sem sinal" também é o som ambiente sem rádio
        N = int(sf_.WIN_S * sf_.FS)
        noise += [sf_.augment_mic(w, rng) for w in noise[:100]]
        noise += [sf_._ambient(rng, N).astype(np.float32) for _ in range(120)]
    Xn = np.vstack([sf_.extract(w) for w in noise])
    Xs.append(Xn); ys += [NOISE_LABEL] * len(Xn); gs += [f"noise{i // 10}" for i in range(len(Xn))]
    augs.append(np.zeros(len(Xn), np.int8))

    X = np.vstack(Xs)
    y = np.array(ys)
    groups = np.array(gs)
    aug = np.concatenate(augs)
    print(f"Matriz: {X.shape[0]} janelas x {X.shape[1]} features, {len(set(y))} classes")

    from sklearn.model_selection import StratifiedGroupKFold
    from sigid_core import fit_model, model_proba

    def fit(Xtr, ytr, n_mlp=1):
        return fit_model(Xtr, ytr, n_mlp=n_mlp)

    def proba(model, Xq):
        return model_proba(model, Xq), model["classes"]

    report = {}
    if not args.no_eval:
        # avaliação honesta: só sinais com >=2 arquivos, arquivos inteiros separados por fold
        files_per = {}
        for g, lab in zip(groups, y):
            files_per.setdefault(lab, set()).add(g)
        ok_cls = {c for c, s in files_per.items() if len(s) >= 2}
        sel = np.isin(y, list(ok_cls))
        Xe, ye, ge, ae = X[sel], y[sel], groups[sel], aug[sel]
        print(f"Validação cruzada (5 folds por arquivo) em {len(ok_cls)} sinais com >=2 arquivos...")
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        w1 = w5 = wn = f1 = f5 = fn = 0
        for k, (tr, te) in enumerate(cv.split(Xe, ye, ge)):
            # treina com tudo que não é do fold de teste (inclui sinais com 1 arquivo)
            test_files = set(ge[te])
            trm = ~np.isin(groups, list(test_files))
            model = fit(X[trm], y[trm])
            te = te[ae[te] == (2 if args.mobile else 0)]  # originais (PC) ou via microfone (mobile)
            Pr, cls = proba(model, Xe[te])
            order = np.argsort(-Pr, 1)
            ci = {c: i for i, c in enumerate(cls)}
            yi = np.array([ci.get(v, -1) for v in ye[te]])
            w1 += (order[:, 0] == yi).sum(); w5 += (order[:, :5] == yi[:, None]).any(1).sum(); wn += len(te)
            for g in set(ge[te]):
                m = ge[te] == g
                pf = Pr[m].mean(0)
                o = np.argsort(-pf)
                t = yi[m][0]
                f1 += o[0] == t; f5 += t in o[:5]; fn += 1
            print(f"  fold {k + 1}: janela top1={w1 / wn:.3f} top5={w5 / wn:.3f} | "
                  f"arquivo top1={f1 / fn:.3f} top5={f5 / fn:.3f}", flush=True)
        report = {"window_top1": w1 / wn, "window_top5": w5 / wn,
                  "file_top1": f1 / fn, "file_top5": f5 / fn, "n_eval_classes": len(ok_cls)}

    mp = MP
    if args.mod_only and os.path.exists(mp):
        print("Reaproveitando o classificador de sinais; retreinando só a modulação...")
        model = joblib.load(mp)
    else:
        print("Treinando modelo final com todos os dados...")
        model = fit(X, y, n_mlp=3)

    # modelo de modulação (rótulos mais genéricos, generaliza melhor)
    mod_y = []
    for lab in y:
        if lab == NOISE_LABEL:
            mod_y.append("RUÍDO")
        else:
            mod_y.append(modulation_label(signals.get(lab, {}).get("fields", {})) or "")
    mod_y = np.array(mod_y)
    counts = {m: (mod_y == m).sum() for m in set(mod_y)}
    msel = np.array([m != "" and counts[m] >= 40 for m in mod_y])
    from sklearn.neural_network import MLPClassifier
    mod_model = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, batch_size=256,
                              early_stopping=True, n_iter_no_change=8, max_iter=150, random_state=1)
    mod_model.fit(np.clip(model["scaler"].transform(X[msel]), -8, 8), mod_y[msel])
    print("Modulações:", ", ".join(sorted(mod_model.classes_)))

    sig_meta = {k: {kk: v.get(kk) for kk in ("fields", "description", "url", "categories")}
                for k, v in signals.items()}
    sig_meta[NOISE_LABEL] = {"fields": {}, "description": "Sem sinal reconhecível: ruído, estática ou silêncio.",
                             "url": "", "categories": []}
    mp = MP
    if not report and os.path.exists(mp):  # --no-eval: mantém a última validação
        try:
            report = joblib.load(mp).get("report", {}) or {}
        except Exception:
            pass
    model.update({"mod": mod_model, "meta": sig_meta, "report": report,
                  "fs": sf_.FS, "win_s": sf_.WIN_S, "feature_version": sf_.FEATURE_VERSION,
                  "n_features": X.shape[1], "trained": time.strftime("%Y-%m-%d %H:%M")})
    model["mobile"] = bool(args.mobile)
    joblib.dump(model, MP, compress=3)
    print("Modelo salvo em", os.path.relpath(MP, ROOT), report)


if __name__ == "__main__":
    main()
