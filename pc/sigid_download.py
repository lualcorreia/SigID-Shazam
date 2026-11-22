"""
Baixa as amostras de áudio do Signal Identification Wiki (sigidwiki.com)
usando a API MediaWiki, e monta data/manifest.json:
  { "signals": {titulo: {categories, fields, url}}, "files": [{path, signal, url}] }

Uso:
  python sigid_download.py              # baixa tudo (identificados)
  python sigid_download.py --include-unidentified
"""
import argparse, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import requests

API = "https://www.sigidwiki.com/api.php"
WIKI = "https://www.sigidwiki.com/wiki/"
UA = "SigID-Shazam/1.0 (pesquisa pessoal de radioamador PP2LA; python-requests)"
AUDIO_EXT = (".mp3", ".ogg", ".oga", ".wav", ".flac", ".opus")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
AUDIO_DIR = os.path.join(DATA, "audio")

S = requests.Session()
S.headers["User-Agent"] = UA
_lock = threading.Lock()
_last = [0.0]


MIN_INTERVAL = 1.2  # segundos entre requisições (o site aplica 429)


def polite_get(url, **kw):
    """GET com espaçamento global entre requisições e tratamento de 429/5xx."""
    for attempt in range(12):
        with _lock:
            dt = time.time() - _last[0]
            if dt < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - dt)
            _last[0] = time.time()
        try:
            r = S.get(url, timeout=120, **kw)
            if r.status_code in (429, 502, 503, 504):
                ra = r.headers.get("Retry-After", "")
                wait = int(ra) if ra.isdigit() else min(30 * (attempt + 1), 300)
                print(f"  [{r.status_code}] aguardando {wait}s...", file=sys.stderr, flush=True)
                r.close()
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  [rede] {e}; tentando de novo...", file=sys.stderr, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"falhou: {url}")


def api(params):
    return polite_get(API, params=dict(params, format="json")).json()


def continue_params(js):
    """Suporta continuação nova ('continue') e antiga ('query-continue')."""
    if "continue" in js:
        return dict(js["continue"])
    if "query-continue" in js:
        out = {}
        for v in js["query-continue"].values():
            out.update(v)
        return out
    return None


def norm_name(n):
    n = n.strip().replace("_", " ")
    n = re.sub(r"^(file|image|media|arquivo)\s*:\s*", "", n, flags=re.I)
    return re.sub(r"\s+", " ", n).lower()


def list_audio_files():
    """Todos os arquivos de áudio do wiki: nome normalizado -> {name, url}."""
    cache = os.path.join(DATA, "filelist.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)
    files = {}
    base = {"action": "query", "list": "allimages", "ailimit": "500",
            "aiprop": "url|mime|size", "continue": ""}
    params = dict(base)
    while True:
        js = api(params)
        for im in js.get("query", {}).get("allimages", []):
            name = im.get("name", "")
            if name.lower().endswith(AUDIO_EXT):
                files[norm_name(name)] = {"name": name, "url": im.get("url"), "size": im.get("size")}
        cont = continue_params(js)
        print(f"  {len(files)} arquivos de áudio até agora", flush=True)
        if not cont:
            break
        params = dict(base)
        params.update(cont)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(files, fh, ensure_ascii=False)
    return files


def list_signal_pages():
    titles = []
    base = {"action": "query", "list": "embeddedin", "eititle": "Template:Signal",
            "einamespace": "0", "eilimit": "500", "continue": ""}
    params = dict(base)
    while True:
        js = api(params)
        titles += [p["title"] for p in js.get("query", {}).get("embeddedin", [])]
        cont = continue_params(js)
        if not cont:
            break
        params = dict(base)
        params.update(cont)
    return sorted(set(titles))


AUDIO_REF = re.compile(r"([^\n|=\[\]{}<>:#]+?\.(?:mp3|ogg|oga|wav|flac|opus))(?=[\s|\]}<]|$)", re.I)


TEMPLATE_FIELD = re.compile(r"^\s*\|\s*([A-Za-z][\w \-]*?)\s*=\s*(.*?)\s*$", re.M)


def clean_wiki(text):
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    return text.strip()


def fetch_signal_meta(titles):
    meta = {}
    titles = sorted(titles)
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        js = api({"action": "query", "titles": "|".join(chunk),
                  "prop": "categories|revisions", "rvprop": "content",
                  "cllimit": "500", "redirects": "1"})
        q = js.get("query", {})
        redirects = {r["from"]: r["to"] for r in q.get("redirects", [])}
        for p in q.get("pages", {}).values():
            t = p.get("title")
            if not t:
                continue
            cats = [c["title"].split(":", 1)[-1] for c in p.get("categories", [])]
            content = ""
            revs = p.get("revisions") or []
            if revs:
                content = revs[0].get("*") or revs[0].get("content") or ""
            fields = {}
            # só o primeiro template (infobox)
            m = re.search(r"\{\{(.*?)\n\}\}", content, re.S)
            block = m.group(1) if m else ""
            for k, v in TEMPLATE_FIELD.findall(block):
                v = clean_wiki(v)
                if v and not re.search(r"\.(png|jpe?g|gif|mp3|ogg|wav)$", v, re.I):
                    fields[k.strip().lower()] = v[:200]
            # descrição: primeiro parágrafo depois do template
            body = content[m.end():] if m else content
            desc = ""
            for para in re.split(r"\n\s*\n", body):
                para = clean_wiki(para)
                if para and not para.startswith(("=", "[", "|", "{", "*", "<")):
                    desc = para[:600]
                    break
            if fields.get("signal description"):
                desc = fields.pop("signal description")
            refs = []
            for r_ in AUDIO_REF.findall(content):
                r_ = r_.strip()
                if r_ not in refs:
                    refs.append(r_)
            meta[t] = {"categories": cats, "fields": fields, "description": desc,
                       "url": WIKI + quote(t.replace(" ", "_"), safe="()_,'-"),
                       "audio_refs": refs}
        for a, b in redirects.items():
            if b in meta:
                meta[a] = meta[b]
        print(f"  metadados {min(i + 40, len(titles))}/{len(titles)}")
    return meta


def is_unidentified(title, m):
    cats = " ".join(m.get("categories", [])).lower()
    tl = title.lower()
    return ("unidentified" in cats or "unidentified" in tl or tl.startswith("unknown")
            or "unknown signal" in cats)


def is_signal_page(m):
    """Página de sinal tem infobox com campos típicos (frequência, modo...)."""
    f = m.get("fields", {})
    keys = " ".join(f.keys())
    return any(k in keys for k in ("frequenc", "mode", "modulation", "bandwidth", "acf", "location"))


def safe(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(" .")[:120]


def download(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, True
    tmp = path + ".part"
    try:
        r = polite_get(url)
        with open(tmp, "wb") as fh:
            fh.write(r.content)
        os.replace(tmp, path)
        return path, True
    except Exception as e:
        print(f"  falhou: {url} ({e})", file=sys.stderr, flush=True)
        return path, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-unidentified", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(AUDIO_DIR, exist_ok=True)

    print("1/3 Listando arquivos de áudio do wiki...", flush=True)
    files = list_audio_files()
    print(f"   {len(files)} áudios no wiki", flush=True)

    print("2/3 Páginas de sinal e metadados...", flush=True)
    titles = list_signal_pages()
    print(f"   {len(titles)} páginas com {{{{Signal}}}}", flush=True)
    meta = fetch_signal_meta(titles)

    def resolve(ref):
        k = norm_name(ref)
        if k in files:
            return k
        words = k.split(" ")  # frase antes do nome do arquivo -> tira palavras do início
        for i in range(1, len(words)):
            kk = " ".join(words[i:])
            if kk in files:
                return kk
        return None

    signals, jobs, manifest_files, taken = {}, [], [], set()
    skipped_unid = 0
    for title in titles:
        m = meta.get(title)
        if not m:
            continue
        if not args.include_unidentified and is_unidentified(title, m):
            skipped_unid += 1
            continue
        keys = [k for k in (resolve(r) for r in m.get("audio_refs", [])) if k]
        keys = [k for k in dict.fromkeys(keys) if k not in taken]
        if not keys:
            continue
        signals[title] = {k_: v for k_, v in m.items() if k_ != "audio_refs"}
        d = os.path.join(AUDIO_DIR, safe(title))
        os.makedirs(d, exist_ok=True)
        for k in keys:
            taken.add(k)
            f = files[k]
            path = os.path.join(d, safe(f["name"]))
            jobs.append((f["url"], path))
            manifest_files.append({"path": os.path.relpath(path, DATA), "signal": title,
                                   "url": f["url"], "file": f["name"]})
    print(f"   {len(jobs)} áudios de {len(signals)} sinais "
          f"(ignorados {skipped_unid} não identificados)", flush=True)

    with open(os.path.join(DATA, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"signals": signals, "files": manifest_files,
                   "created": time.strftime("%Y-%m-%d %H:%M")}, fh, ensure_ascii=False, indent=1)
    print("   manifesto salvo em data/manifest.json", flush=True)

    print("3/3 Baixando áudios...", flush=True)
    ok = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (path, good) in enumerate(ex.map(download, jobs), 1):
            ok += good
            if i % 25 == 0 or i == len(jobs):
                print(f"   {i}/{len(jobs)} ({ok} ok)", flush=True)
    print("DOWNLOAD CONCLUIDO", flush=True)


if __name__ == "__main__":
    main()
