"""Teste ao vivo: toca um áudio no 'CABLE Input' e o app escuta pelo 'CABLE Output'.
Uso: python tests_live_cable.py arquivo.mp3 [segundos]"""
import sys, os, threading, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np, sounddevice as sd, soundfile as sf
from PyQt6 import QtCore, QtWidgets
import app as A

path = sys.argv[1]
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 14
x, sr = sf.read(path, dtype="float32", always_2d=True)
x = x.mean(1)
x = np.tile(x, int(np.ceil(dur * sr / len(x))))[: int(dur * sr)]
out_dev = next(i for i, d in enumerate(sd.query_devices())
               if "cable input" in d["name"].lower() and d["max_output_channels"] > 0
               and sd.query_hostapis(d["hostapi"])["name"] == "MME")

qa = QtWidgets.QApplication(sys.argv)
w = A.Main()
mme = [k for k in range(w.dev.count()) if "cable output" in w.dev.itemText(k).lower() and "MME" in w.dev.itemText(k)]
if mme:
    w.dev.setCurrentIndex(mme[0])
print("escutando:", w.dev.currentText())
if os.environ.get("QRG_MHZ"):  # liga a lista QRGBuscador nessa frequência
    w.freq.setValue(float(os.environ["QRG_MHZ"]))
    w.use_qrg.setChecked(True)
    w.qrg_boost.setCurrentIndex(int(os.environ.get("QRG_PESO", "1")))
w.btn_listen.setChecked(True)
threading.Thread(target=lambda: (time.sleep(0.5), sd.play(x / (np.abs(x).max() + 1e-9) * 0.5, sr,
                                                          device=out_dev, blocking=True)), daemon=True).start()


def end():
    print("nível:", w.level.value(), "dBFS")
    print("melhor:", w.best.text())
    print(w.best_sub.text())
    print(w.mod_label.text())
    for i in range(w.table.topLevelItemCount()):
        it = w.table.topLevelItem(i)
        print("  ", it.text(0), it.text(1), it.text(2).split()[-1])
    print("histórico:", [w.history.item(i).text() for i in range(w.history.count())])
    print(w.qrg_status.text(), "|", w.tabs.tabText(1))
    for i in range(min(3, w.qrg_table.topLevelItemCount())):
        it = w.qrg_table.topLevelItem(i)
        print("   lista:", " | ".join(it.text(c) for c in range(9)))
    w.close(); qa.quit()


QtCore.QTimer.singleShot(int(dur * 1000), end)
qa.exec()
