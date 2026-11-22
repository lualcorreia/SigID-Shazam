"""
Dicas de QRG a partir das listas do QRGBuscador (github.com/lualcorreia/QRGBuscador):
  * ILGADATA.TXT  — base ILG (dentro de ILGADATA.zip): estação, horário, emissão, protocolo, indicativo
  * programadas.txt — transmissões programadas (marítimo, aeronáutico, meteorológico, sinais de tempo)

A lista NÃO decide nada: ela só dá um empurrão (peso >= 1) nos sinais que costumam
aparecer perto da frequência sintonizada. Quem decide continua sendo o áudio.

  python qrg_hints.py --update        # baixa as listas do GitHub
  python qrg_hints.py 8414.5 [tol]    # mostra o que a lista sugere perto de 8414,5 kHz
"""
import os, re, sys, bisect, pickle, zipfile, datetime
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(ROOT, "data", "qrg")
ZIP = os.path.join(QDIR, "ILGADATA.zip")
PROG = os.path.join(QDIR, "programadas.txt")
CACHE = os.path.join(QDIR, "qrg_index.pkl")
REPO_RAW = "https://raw.githubusercontent.com/lualcorreia/QRGBuscador/HEAD/"
CACHE_VERSION = 2


# ---------------------------------------------------------------- download
def update_lists():
    import requests
    os.makedirs(QDIR, exist_ok=True)
    for name, dest in (("ILGADATA.zip", ZIP), ("programadas.txt", PROG)):
        r = requests.get(REPO_RAW + name, timeout=120, headers={"User-Agent": "SigID-Shazam"})
        r.raise_for_status()
        with open(dest + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dest + ".part", dest)
        print(f"  {name}: {len(r.content) / 1e6:.1f} MB")
    if os.path.exists(CACHE):
        os.replace(CACHE, CACHE + ".old")


# ---------------------------------------------------------------- ILGA
RF = re.compile(r"^\s*(\d{1,5}\.\d{2})")
COORD = re.compile(r"(\d{3}) (\d{2}) (\d{2})([EW]),(\d{2}) (\d{2}) (\d{2})([NS])")
TAIL = re.compile(r"(?<![A-Z0-9])([CX])(\d{4})(.*?)\s*\d{1,5}\.\d{2}\d{4} \d{4}")
EMIS = re.compile(r"\b(USB|LSB|ISB|AM|DRM|FM|CW)\s?((?:[A-Z]\d[A-Z])+)")
TIME = re.compile(r"(\d{4})-(\d{4})")


def _dms(d, m, s, h):
    v = int(d) + int(m) / 60 + int(s) / 3600
    return -v if h in "WS" else v


def parse_ilga(text):
    out = []
    for l in text.splitlines():
        m = RF.match(l)
        if not m:
            continue
        st = l[8:38].strip()
        if not st or "###" in st or "STATION" in st:
            continue
        tm = TIME.match(l[38:47])
        em = EMIS.search(l[180:215])
        e = {"f": float(m.group(1)), "station": st, "call": "", "kind": l[54:72].strip(),
             "remark": "", "mode": em.group(1) if em else "",
             "emis": " ".join(re.findall(r"[A-Z]\d[A-Z]", em.group(2))) if em else "",
             "t0": int(tm.group(1)) if tm else None, "t1": int(tm.group(2)) if tm else None,
             "days": re.sub(r"\D", "", l[47:54]), "loc": l[130:156].strip(),
             "lat": None, "lon": None, "src": "ILGA", "section": ""}
        cm = COORD.search(l, 200)
        if cm:
            g = cm.groups()
            e["lon"], e["lat"] = _dms(*g[0:4]), _dms(*g[4:8])
            after = l[cm.end():]
            tt = TAIL.search(after)
            if tt:
                e["remark"] = after[:tt.start()].strip()
                e["call"] = tt.group(3).strip()[:20]
            else:
                e["remark"] = after[:50].strip()
        out.append(e)
    return out


# ---------------------------------------------------------------- programadas.txt
def _freq_token(s):
    s = s.strip().rstrip("#*").strip()
    m = re.fullmatch(r"(\d{1,2})\.(\d{3})\.(\d)", s)  # 4.209.5 -> 4209,5 kHz
    if m:
        return int(m.group(1)) * 1000 + int(m.group(2)) + int(m.group(3)) / 10
    if re.fullmatch(r"\d{1,2}\.\d{1,3}", s) and float(s) < 30:  # MHz
        return float(s) * 1000
    if re.fullmatch(r"\d{3,5}(\.\d+)?", s) and 100 <= float(s) <= 30000:  # kHz
        return float(s)
    return None


PROG_SECTIONS = ("marine band broadcasts", "hf voice", "dsc", "hf radio-telex broadcasts", "volmet",
                 "hfdl", "afi", "rtty", "mx radiofax", "sinal de tempo")


def parse_programadas(text):
    out, sec, band, cur = [], None, None, None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        hm = re.fullmatch(r"(\d+(?:\.\d+)?)\s+MHz", s, re.I)
        if hm:
            band = float(hm.group(1)) * 1000
            continue
        if "\t" not in s and not raw[:1].isspace() and not s[:1].isdigit():
            low = s.lower()
            sec = next((k for k in PROG_SECTIONS if low.startswith(k)), None)
            band = cur = None
            continue
        if sec is None or s.lower().startswith("freq\t"):
            continue
        f = [x.strip() for x in raw.split("\t")]
        call, station = "", ""
        for i in range(len(f) - 1):
            if re.fullmatch(r"[A-Z0-9]{2,6}", f[i]) and re.fullmatch(r"[A-Z]{3}", f[i + 1]) and not f[i].isdigit():
                call = f[i]
                station = f[i + 2] if i + 2 < len(f) else ""
                break
        remark = re.sub(r"\s+", " ", " ".join(x for x in f[1:] if x and x not in (call, station)))[:90]
        if sec in ("dsc", "afi"):
            freqs = [v for v in (_freq_token(x) for x in f if "." in x) if v]
        elif sec == "sinal de tempo":
            freqs = [band] if band else []
        else:
            v = _freq_token(f[0]) if f[0] else None
            if v:
                cur = v
            freqs = [cur] if cur else []
        for v in freqs:
            out.append({"f": v, "station": station or call, "call": call, "kind": "", "remark": remark,
                        "mode": "", "emis": "", "t0": None, "t1": None, "days": "", "loc": station,
                        "lat": None, "lon": None, "src": "Programadas", "section": sec})
    return out


def load_entries():
    srcs = [p for p in (ZIP, PROG) if os.path.exists(p)]
    if not srcs:
        return []
    stamp = tuple(os.path.getmtime(p) for p in srcs)
    if os.path.exists(CACHE):
        try:
            with open(CACHE, "rb") as fh:
                c = pickle.load(fh)
            if c.get("v") == CACHE_VERSION and c.get("stamp") == stamp:
                return c["entries"]
        except Exception:
            pass
    entries = []
    if os.path.exists(ZIP):
        with zipfile.ZipFile(ZIP) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
            entries += parse_ilga(z.read(name).decode("latin-1"))
    if os.path.exists(PROG):
        entries += parse_programadas(open(PROG, encoding="utf-8", errors="replace").read())
    entries.sort(key=lambda e: e["f"])
    with open(CACHE, "wb") as fh:
        pickle.dump({"v": CACHE_VERSION, "stamp": stamp, "entries": entries}, fh)
    return entries


# ---------------------------------------------------------------- lista -> classes do sigidwiki
class _has:
    """Regra 'todas as classes cujo nome contém sub'."""
    def __init__(self, sub, s=0.7):
        self.sub, self.s = sub.lower(), s

    def pairs(self, classes):
        return [(c, self.s) for c in classes if self.sub in c.lower()]


def _pactor(m):
    n = {"I": "I", "1": "I", "II": "II", "2": "II", "III": "III", "3": "III", "IV": "IV", "4": "IV"}
    return [(f"PACTOR {n[m.group(1).upper()]}", 1.0)]


# (regex no texto da entrada, lista fixa | função(match) | função(classes))
RULES = [
    (r"STANAG\s*-?\s*(\d{4})", lambda m: [(f"STANAG {m.group(1)}", 1.0)]),
    (r"188-110", [("MIL-STD-188-110 Serial", 1.0), ("MIL-STD-188-110 Appendix B 39-Tone", .5),
                  ("MIL-STD-188-110 Appendix A 16-Tone", .5), ("STANAG 4539", .5)]),
    (r"188-141\s*B|\b3G\b", [("Automatic Link Establishment (3G ALE ARCS)", 1.0)]),
    (r"WALE|\b4G\b|188-141\s*D", [("Automatic Link Establishment (4G ALE WALE)", 1.0)]),
    (r"\bALE\b|188-141", [("Automatic Link Establishment (2G ALE)", 1.0),
                          ("Automatic Link Establishment (3G ALE ARCS)", .3)]),
    (r"PACTOR\s*-?\s*(IV|III|II|I|4|3|2|1)\b", _pactor),
    (r"PACTOR", _has("PACTOR")),
    (r"SITOR\s*-?\s*A|AMTOR", [("SITOR-A", 1.0)]),
    (r"SITOR\s*-?\s*B|NAVTEX", [("SITOR-B", 1.0)]),
    (r"SITOR", [("SITOR-A", .6), ("SITOR-B", .6)]),
    (r"\bDSC\b|SELECTIVE CALL", [("GMDSS Digital Selective Calling", 1.0)]),
    (r"HFDL", [("High Frequency Data Link (HFDL)", 1.0)]),
    (r"DGPS", [("Differential Global Positioning System (DGPS)", 1.0)]),
    (r"\bFAX\b|WEFAX|RADIOFAX|\bF3C\b", [("WEFAX", 1.0)]),
    (r"MORSE|\bA1A\b", [("Morse Code (CW)", 1.0)]),
    (r"RTTY|BAUDOT|\b\d{2,3}\s*BD\s*/\s*\d{2,4}\b|\b(45|50|75)\s*/\s*(170|425|450|850)\b",
     [("Radio Teletype (RTTY)", 1.0)]),
    (r"TELEX|\bF1B\b", [("Radio Teletype (RTTY)", .6), ("SITOR-B", .4), ("SITOR-A", .3)]),
    (r"LINK\s*-?\s*11", [("Link-11", 1.0)]),
    (r"CODAN", [("Codan Data Modem", 1.0)]),
    (r"THALES|SYSTEME 3000|SKYHOPPER", _has("Thales")),
    (r"CLOVER", _has("CLOVER")),
    (r"\bDRM\b|\bDRMD\b", [("Digital Radio Mondiale (DRM)", 1.0)]),
    (r"VOLMET|\bJ3E\b|\bVOICE\b", [("Single Sideband Voice", .6)]),
    (r"SELCAL", [("ICAO Selcal", .8)]),
    (r"\bOTH\b|RADAR", _has("OTH")),
    (r"TIME SIGNAL|TIME STANDARD|TIME ANNOUNCE", [("UTC (Coordinated Universal Time) Time Standard", .8)]),
]
GENERIC_RULES = {
    r"RTTY|BAUDOT|\b\d{2,3}\s*BD\s*/\s*\d{2,4}\b|\b(45|50|75)\s*/\s*(170|425|450|850)\b",
    r"TELEX|\bF1B\b", r"VOLMET|\bJ3E\b|\bVOICE\b", r"\bOTH\b|RADAR",
    r"TIME SIGNAL|TIME STANDARD|TIME ANNOUNCE",
}
SECTION_RULES = {
    "marine band broadcasts": [("Single Sideband Voice", .6)],
    "hf voice": [("Single Sideband Voice", .8)],
    "dsc": [("GMDSS Digital Selective Calling", 1.0)],
    "hf radio-telex broadcasts": [("SITOR-B", .9), ("Radio Teletype (RTTY)", .3)],
    "volmet": [("Single Sideband Voice", .9)],
    "hfdl": [("High Frequency Data Link (HFDL)", 1.0)],
    "afi": [("Single Sideband Voice", .6), ("ICAO Selcal", .4)],
    "rtty": [("Radio Teletype (RTTY)", 1.0)],
    "mx radiofax": [("WEFAX", 1.0)],
    "sinal de tempo": [("UTC (Coordinated Universal Time) Time Standard", .8)],
}
SECTION_NAMES = {"marine band broadcasts": "Marítimo (voz)", "hf voice": "Voz HF", "dsc": "DSC",
                 "hf radio-telex broadcasts": "Radiotelex", "volmet": "VOLMET", "hfdl": "HFDL",
                 "afi": "Aeronáutico AFI", "rtty": "RTTY meteo", "mx radiofax": "Radiofax",
                 "sinal de tempo": "Sinal de tempo"}
STOP = {"VOICE", "DATA", "RADAR", "BEACON", "MODEM", "LINK", "SIGNAL", "BROADCAST", "FSK", "PSK", "OFDM",
        "MFSK", "AM", "FM", "NFM", "CW", "USB", "LSB", "QPSK", "BPSK", "8PSK", "ASK", "OOK", "MSK", "QAM",
        "RADIO", "UNKNOWN", "SERIAL", "NAVY", "MILITARY", "TONE", "STANDARD", "HF", "VHF", "UHF", "TIME"}
FAMILY_TOKENS = [("MFSK", r"MFSK|\b\d+\s*TONE"), ("PSK", r"PSK"), ("OFDM", r"OFDM"), ("QAM", r"QAM"),
                 ("FSK", r"(?<!M)FSK|\bF1B\b"), ("OOK/ASK", r"\bA1A\b|MORSE"), ("AM", r"\bA3E\b|\bJ3E\b")]


def _family_of(modulation):
    m = (modulation or "").upper()
    for fam, rx in FAMILY_TOKENS:
        if re.search(rx, m):
            return fam
    return None


def is_active(e, now):
    if e["t0"] is None:
        return None
    hm = now.hour * 100 + now.minute
    t0, t1 = e["t0"], e["t1"]
    if t0 == t1 or (t0 == 0 and t1 >= 2400):
        on = True
    elif t0 < t1:
        on = t0 <= hm < t1
    else:
        on = hm >= t0 or hm < t1
    d = str((now.weekday() + 1) % 7 + 1)  # ILGA: 1=domingo
    if e["days"] and d not in e["days"]:
        on = False
    return on


class QRGHints:
    def __init__(self, classes, meta, entries=None):
        self.entries = load_entries() if entries is None else entries
        self.freqs = [e["f"] for e in self.entries]
        self.classes = list(classes)
        self.idx = {c.lower(): i for i, c in enumerate(self.classes)}
        self.fam = [_family_of(meta.get(c, {}).get("fields", {}).get("modulation", "")) for c in self.classes]
        from sigid_core import parse_freqs
        self.ranges = [parse_freqs(meta.get(c, {}).get("fields", {}).get("frequencies", "")) for c in self.classes]
        # apelidos genéricos: abreviação entre parênteses e o título, como palavras inteiras
        alias = {}
        for i, c in enumerate(self.classes):
            cands = re.findall(r"\(([A-Z0-9][A-Z0-9\- ]{1,10})\)", c)
            base = re.sub(r"\s*\(.*?\)\s*", " ", c).strip().upper()
            cands.append(base)
            for a in cands:
                a = a.strip().upper()
                if len(a) >= 3 and a not in STOP and not a.isdigit():
                    alias.setdefault(a, set()).add(i)
        self.alias = alias
        keys = sorted(alias, key=len, reverse=True)
        self.alias_rx = re.compile(r"(?<![A-Z0-9])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Z0-9])") \
            if keys else None
        self.rules = [(re.compile(rx), act, rx in GENERIC_RULES) for rx, act in RULES]
        self._cache = {}

    def __len__(self):
        return len(self.entries)

    def _resolve(self, pairs, out):
        for name, s in pairs:
            i = self.idx.get(name.lower())
            if i is not None:
                out[i] = max(out.get(i, 0), s)

    def match(self, e):
        """Classes do modelo compatíveis com a entrada da lista -> {índice: força 0..1}."""
        key = (e["station"], e["kind"], e["remark"], e["emis"], e["section"], e["call"])
        if key in self._cache:
            return self._cache[key]
        text = " ".join([e["station"], e["kind"], e["remark"], e["mode"], e["emis"]]).upper()
        out, generic = {}, {}
        for rx, act, is_generic in self.rules:
            m = rx.search(text)
            if m:
                if isinstance(act, list):
                    pairs = act
                elif isinstance(act, _has):
                    pairs = act.pairs(self.classes)
                else:
                    pairs = act(m)
                self._resolve(pairs, generic if is_generic else out)
        # protocolo concreto reconhecido -> pistas genéricas (TELEX, baud/shift, voz) valem pouco
        k = 0.3 if out else 1.0
        for i, s in generic.items():
            out[i] = max(out.get(i, 0), s * k)
        if e["section"] in SECTION_RULES:
            self._resolve(SECTION_RULES[e["section"]], out)
        if self.alias_rx:
            for a in set(self.alias_rx.findall(text)):
                for i in self.alias[a]:
                    out[i] = max(out.get(i, 0), 0.8)
        if e["call"]:
            i = self.idx.get(e["call"].lower())
            if i is not None:
                out[i] = 1.0
        fam = {}
        for f, rx in FAMILY_TOKENS:
            if re.search(rx, text):
                fam[f] = 1.0
        res = (out, fam)
        self._cache[key] = res
        return res

    def near(self, f_khz, tol_khz):
        a = bisect.bisect_left(self.freqs, f_khz - tol_khz)
        b = bisect.bisect_right(self.freqs, f_khz + tol_khz)
        return self.entries[a:b]

    def weights(self, freq_hz, tol_khz=2.0, boost=2.0, use_time=True, now=None):
        """Retorna (pesos >= 1 por classe, linhas para exibir)."""
        w = np.ones(len(self.classes))
        if not freq_hz or not self.entries:
            return w, []
        now = now or datetime.datetime.now(datetime.timezone.utc)
        f = freq_hz / 1e3
        score = np.zeros(len(self.classes))
        famscore = {}
        rows = []
        for e in self.near(f, tol_khz):
            act = is_active(e, now) if use_time else None
            tw = 0.7 if act is None else (1.0 if act else 0.35)
            d = abs(e["f"] - f)
            dw = 1.0 - 0.5 * d / max(tol_khz, 1e-6)
            cls, fam = self.match(e)
            for i, s in cls.items():
                score[i] = max(score[i], s * tw * dw)
            for k, s in fam.items():
                famscore[k] = max(famscore.get(k, 0), s * tw * dw)
            order = sorted(cls, key=lambda i: -cls[i])
            rows.append({"e": e, "d": d, "active": act,
                         "classes": [self.classes[i] for i in order],
                         "strong": [self.classes[i] for i in order if cls[i] >= 0.6]})
        w += boost * score
        for i, fam in enumerate(self.fam):
            # família de modulação só reforça sinais que o sigidwiki põe nessa faixa
            if fam and fam in famscore and any(a <= freq_hz <= b for a, b in self.ranges[i]):
                w[i] *= 1 + 0.25 * famscore[fam]
        rows.sort(key=lambda r: (r["active"] is False, r["d"]))
        return w, rows


def describe(e):
    if e["src"] == "Programadas":
        return SECTION_NAMES.get(e["section"], e["section"]) + (f" · {e['remark']}" if e["remark"] else "")
    parts = [e["kind"].replace("DATA:", "").strip(), e["remark"]]
    return " · ".join(p for p in parts if p)


def main():
    if "--update" in sys.argv:
        print("Baixando listas do QRGBuscador...")
        update_lists()
    entries = load_entries()
    print(f"{len(entries)} entradas na lista ({sum(e['src'] == 'ILGA' for e in entries)} ILGA, "
          f"{sum(e['src'] == 'Programadas' for e in entries)} programadas)")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        return
    f = float(args[0].replace(",", "."))
    tol = float(args[1]) if len(args) > 1 else 2.0
    from sigid_core import Identifier
    ident = Identifier()
    q = QRGHints(ident.classes, ident.meta, entries)
    w, rows = q.weights(f * 1e3, tol)
    print(f"\n{len(rows)} entradas em {f} ± {tol} kHz:")
    for r in rows[:25]:
        e = r["e"]
        st = {True: "agora", False: "fora do horário", None: ""}[r["active"]]
        print(f"  {e['f']:9.2f}  {e['station'][:28]:28s} {e['call'][:10]:10s} {describe(e)[:45]:45s} "
              f"{e['emis']:8s} {st:15s} -> {', '.join(r['classes'][:3])}")
    o = np.argsort(-w)[:8]
    print("\nPesos aplicados (os demais = 1,00×):")
    for i in o:
        if w[i] > 1.001:
            print(f"  {w[i]:.2f}×  {ident.classes[i]}")


if __name__ == "__main__":
    main()
