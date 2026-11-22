"""Identifica arquivos de áudio pela linha de comando.

  python identify.py gravacao.wav [outro.mp3 ...]
"""
import sys
from sigid_core import Identifier


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    ident = Identifier()
    for path in sys.argv[1:]:
        p, pm, _ = ident.identify_file(path)
        print(f"\n== {path}")
        mods = ", ".join(f"{m} {v:.0%}" for m, v in ident.top_mod(pm))
        print(f"   modulação provável: {mods}")
        for i, (lab, v) in enumerate(ident.top(p), 1):
            f = ident.info(lab).get("fields", {})
            extra = " | ".join(x for x in (f.get("mode"), f.get("modulation"), f.get("frequencies")) if x)
            print(f"   {i}. {lab:55s} {v:6.1%}   {extra}")


if __name__ == "__main__":
    main()
