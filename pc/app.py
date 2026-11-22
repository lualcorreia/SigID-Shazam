"""
SigID Shazam — identificador de sinais de rádio pelo áudio demodulado.

Escolha o dispositivo de áudio por onde chega o som do SDR (ex.: "CABLE Output"
do VB-Audio Virtual Cable, ou "Mixagem estéreo"), clique em Ouvir e aguarde
alguns segundos. O app compara o áudio com o modelo treinado nas amostras do
Signal Identification Wiki (sigidwiki.com).
"""
import os, sys, queue, threading, time, datetime, webbrowser
import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

import sigfeatures as sf_
from sigid_core import Identifier, Priors, MODEL_PATH
import qrg_hints as QH

ROOT = os.path.dirname(os.path.abspath(__file__))
FROZEN = getattr(sys, "frozen", False)  # executável do instalador (PyInstaller)
REC_DIR = (os.path.join(os.path.expanduser("~"), "Documents", "SigID-Shazam", "gravacoes") if FROZEN
           else os.path.join(ROOT, "recordings"))
VERSION = "1.0.0"
FMAX = 8000          # Hz exibidos no waterfall
WF_ROWS = 400        # linhas de histórico do waterfall (~20 s)
SILENCE_DBFS = -72
pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


# ---------------------------------------------------------------- áudio
class AudioEngine:
    def __init__(self):
        self.q = queue.Queue(maxsize=400)
        self.stream = None
        self.sr = 48000
        self.channels = 1

    def start(self, device, sr):
        self.stop()
        info = sd.query_devices(device)
        last = None
        for ch in (1, min(2, info["max_input_channels"]), info["max_input_channels"]):
            try:
                self.stream = sd.InputStream(device=device, channels=ch, samplerate=sr,
                                             blocksize=int(sr * 0.04), dtype="float32",
                                             callback=self._cb)
                self.stream.start()
                self.sr, self.channels = sr, ch
                return
            except Exception as e:
                last = e
                self.stream = None
        raise last

    def _cb(self, indata, frames, t, status):
        try:
            self.q.put_nowait(indata.mean(axis=1).copy())
        except queue.Full:
            pass

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
            self.stream = None
        while not self.q.empty():
            self.q.get_nowait()


# ---------------------------------------------------------------- classificação em thread
class Worker(QtCore.QObject):
    result = QtCore.pyqtSignal(object, object, float)
    file_result = QtCore.pyqtSignal(str, object, object, object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, ident):
        super().__init__()
        self.ident = ident
        self.q = queue.Queue(maxsize=1)
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, x16):
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put_nowait(x16)

    def _loop(self):
        while True:
            x = self.q.get()
            t = time.time()
            try:
                p, pm = self.ident.proba(x)
                self.result.emit(p, pm, time.time() - t)
            except Exception as e:
                self.error.emit(str(e))

    def analyze_file(self, path):
        def run():
            try:
                p, pm, P = self.ident.identify_file(path)
                self.file_result.emit(path, p, pm, P)
            except Exception as e:
                self.error.emit(f"{os.path.basename(path)}: {e}")
        threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- janela
class Main(QtWidgets.QMainWindow):
    qrg_loaded = QtCore.pyqtSignal(object, str)

    def __init__(self):
        super().__init__()
        self.qrg = None
        self.qrg_w = None
        self.qrg_rows = []
        self.setWindowTitle("SigID Shazam — identificação de sinais de rádio")
        self.resize(1380, 820)
        self.audio = AudioEngine()
        self.ident = None
        self.worker = None
        self.priors = None
        self.p_s = None
        self.p_m = None
        self.hold = False
        self.ring = np.zeros(0, np.float32)
        self.last_best = None
        self._build_ui()
        self._load_model()
        self._fill_devices()
        self.timer = QtCore.QTimer(self, interval=50, timeout=self._tick)
        self.tick_n = 0

    # ---------- UI
    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        bar = QtWidgets.QHBoxLayout()
        root.addLayout(bar)
        bar.addWidget(QtWidgets.QLabel("Entrada de áudio:"))
        self.dev = QtWidgets.QComboBox(minimumWidth=380)
        bar.addWidget(self.dev, 2)
        b = QtWidgets.QPushButton("↻")
        b.setToolTip("Atualizar lista de dispositivos")
        b.clicked.connect(self._fill_devices)
        bar.addWidget(b)
        self.btn_listen = QtWidgets.QPushButton("▶  Ouvir")
        self.btn_listen.setCheckable(True)
        self.btn_listen.setMinimumWidth(110)
        self.btn_listen.toggled.connect(self._toggle_listen)
        bar.addWidget(self.btn_listen)
        self.btn_file = QtWidgets.QPushButton("Analisar arquivo…")
        self.btn_file.clicked.connect(self._open_file)
        bar.addWidget(self.btn_file)
        self.btn_rec = QtWidgets.QPushButton("● Salvar últimos 30 s")
        self.btn_rec.clicked.connect(self._save_recording)
        bar.addWidget(self.btn_rec)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Nível:"))
        self.level = QtWidgets.QProgressBar(minimum=-90, maximum=0, value=-90, textVisible=True)
        self.level.setFormat("%v dBFS")
        self.level.setFixedWidth(150)
        bar.addWidget(self.level)

        hint = QtWidgets.QHBoxLayout()
        root.addLayout(hint)
        self.use_freq = QtWidgets.QCheckBox("Dica de frequência sintonizada (MHz):")
        hint.addWidget(self.use_freq)
        self.freq = QtWidgets.QDoubleSpinBox(decimals=4, minimum=0.001, maximum=30000, value=7.1)
        self.freq.setFixedWidth(120)
        hint.addWidget(self.freq)
        hint.addSpacing(20)
        hint.addWidget(QtWidgets.QLabel("Modo de demodulação no SDR:"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["(não usar)", "USB", "LSB", "CW", "AM", "NFM", "WFM"])
        hint.addWidget(self.mode)
        hint.addSpacing(20)
        hint.addWidget(QtWidgets.QLabel("Suavização:"))
        self.smooth = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, minimum=1, maximum=10, value=6)
        self.smooth.setFixedWidth(120)
        self.smooth.setToolTip("Quanto maior, mais o resultado considera os segundos anteriores")
        hint.addWidget(self.smooth)
        self.btn_hold = QtWidgets.QPushButton("Congelar")
        self.btn_hold.setCheckable(True)
        self.btn_hold.toggled.connect(lambda v: setattr(self, "hold", v))
        hint.addWidget(self.btn_hold)
        b = QtWidgets.QPushButton("Limpar")
        b.clicked.connect(self._reset)
        hint.addWidget(b)
        hint.addStretch(1)
        for w in (self.use_freq, self.freq, self.mode):
            (w.toggled if isinstance(w, QtWidgets.QCheckBox) else
             w.valueChanged if isinstance(w, QtWidgets.QDoubleSpinBox) else
             w.currentIndexChanged).connect(self._redraw_results)

        # lista QRGBuscador (ILGA + programadas): só um empurrão no peso, o áudio decide
        hint2 = QtWidgets.QHBoxLayout()
        root.addLayout(hint2)
        self.use_qrg = QtWidgets.QCheckBox("Lista QRGBuscador (ILGA + programadas) na frequência acima")
        self.use_qrg.setToolTip("Dá mais peso aos modos/protocolos que a lista registra perto da QRG.\n"
                                "Não elimina nada: se o áudio não bater, a lista não decide.")
        hint2.addWidget(self.use_qrg)
        hint2.addWidget(QtWidgets.QLabel("±"))
        self.qrg_tol = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0.1, maximum=25, value=2.0, suffix=" kHz")
        self.qrg_tol.setFixedWidth(90)
        hint2.addWidget(self.qrg_tol)
        hint2.addWidget(QtWidgets.QLabel("Peso:"))
        self.qrg_boost = QtWidgets.QComboBox()
        self.qrg_boost.addItems(["leve (até ~1,5×)", "médio (até ~3×)", "forte (até ~5×)"])
        self.qrg_boost.setCurrentIndex(1)
        hint2.addWidget(self.qrg_boost)
        self.qrg_time = QtWidgets.QCheckBox("considerar horário UTC")
        self.qrg_time.setChecked(True)
        self.qrg_time.setToolTip("Entradas fora do horário/dia de transmissão da ILGA pesam menos")
        hint2.addWidget(self.qrg_time)
        b = QtWidgets.QPushButton("Atualizar lista do GitHub")
        b.clicked.connect(self._update_qrg_lists)
        hint2.addWidget(b)
        self.qrg_status = QtWidgets.QLabel("lista: carregando…")
        hint2.addWidget(self.qrg_status)
        hint2.addStretch(1)
        for w in (self.use_qrg, self.qrg_time):
            w.toggled.connect(self._refresh_qrg)
        self.freq.valueChanged.connect(self._refresh_qrg)
        self.qrg_tol.valueChanged.connect(self._refresh_qrg)
        self.qrg_boost.currentIndexChanged.connect(self._refresh_qrg)
        self.qrg_loaded.connect(self._on_qrg_loaded)
        self.qrg_timer = QtCore.QTimer(self, interval=60000, timeout=self._refresh_qrg)
        self.qrg_timer.start()

        split = QtWidgets.QSplitter()
        root.addWidget(split, 1)

        # esquerda: espectro + waterfall
        left = pg.GraphicsLayoutWidget()
        split.addWidget(left)
        self.spec_plot = left.addPlot(row=0, col=0)
        self.spec_plot.setLabel("left", "dB")
        self.spec_plot.setXRange(0, FMAX, padding=0)
        self.spec_plot.setYRange(-110, -10)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen("#4fc3f7", width=1))
        self.wf_plot = left.addPlot(row=1, col=0)
        self.wf_plot.setLabel("bottom", "Frequência de áudio", "Hz")
        self.wf_plot.setXLink(self.spec_plot)
        self.wf_plot.invertY(True)
        self.wf_plot.hideAxis("left")
        left.ci.layout.setRowStretchFactor(1, 3)
        self.wf_img = pg.ImageItem()
        self.wf_img.setLookupTable(pg.colormap.get("inferno").getLookupTable(nPts=256))
        self.wf_plot.addItem(self.wf_img)
        self.wf = np.full((WF_ROWS, 512), -120.0, np.float32)
        self.wf_img.setImage(self.wf, levels=(-110, -30))
        self.wf_img.setRect(QtCore.QRectF(0, 0, FMAX, WF_ROWS))

        # direita: resultados
        right = QtWidgets.QWidget()
        split.addWidget(right)
        split.setSizes([800, 580])
        rl = QtWidgets.QVBoxLayout(right)
        self.best = QtWidgets.QLabel("—")
        f = self.best.font(); f.setPointSize(17); f.setBold(True)
        self.best.setFont(f)
        self.best.setWordWrap(True)
        rl.addWidget(self.best)
        self.best_sub = QtWidgets.QLabel("Escolha a entrada de áudio e clique em Ouvir.")
        self.best_sub.setWordWrap(True)
        rl.addWidget(self.best_sub)
        self.mod_label = QtWidgets.QLabel("")
        rl.addWidget(self.mod_label)

        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["#", "Sinal (sigidwiki)", "Confiança", "Modo", "Modulação"])
        self.table.setRootIsDecorated(False)
        self.table.setColumnWidth(0, 28)
        self.table.setColumnWidth(1, 270)
        self.table.setColumnWidth(2, 110)
        self.table.setColumnWidth(3, 60)
        self.table.itemSelectionChanged.connect(self._show_details)
        self.table.itemDoubleClicked.connect(lambda *_: self._open_wiki())
        rl.addWidget(self.table, 2)

        self.tabs = QtWidgets.QTabWidget()
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenExternalLinks(True)
        self.tabs.addTab(self.details, "Detalhes do sinal")
        self.qrg_table = QtWidgets.QTreeWidget()
        self.qrg_table.setHeaderLabels(["kHz", "Estação", "Indicativo", "Tipo / protocolo", "Emissão",
                                        "UTC", "Agora", "Fonte", "Combina"])
        self.qrg_table.setRootIsDecorated(False)
        for i, wdt in enumerate((70, 170, 80, 220, 60, 80, 50, 80, 60)):
            self.qrg_table.setColumnWidth(i, wdt)
        self.tabs.addTab(self.qrg_table, "QRG (lista)")
        rl.addWidget(self.tabs, 2)
        b = QtWidgets.QPushButton("Abrir página do sinal no sigidwiki")
        b.clicked.connect(self._open_wiki)
        rl.addWidget(b)

        rl.addWidget(QtWidgets.QLabel("Histórico de detecções:"))
        self.history = QtWidgets.QListWidget()
        rl.addWidget(self.history, 1)

        self.status = self.statusBar()

    # ---------- modelo / dispositivos
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.best.setText("Modelo não encontrado")
            self.best_sub.setText("Rode primeiro:  python sigid_download.py  e  python train.py")
            return
        try:
            self.ident = Identifier()
        except Exception as e:
            self.best.setText("Erro ao carregar modelo")
            self.best_sub.setText(str(e))
            return
        self.priors = Priors(self.ident)
        self.worker = Worker(self.ident)
        self.worker.result.connect(self._on_result)
        self.worker.file_result.connect(self._on_file_result)
        self.worker.error.connect(lambda m: self.status.showMessage("Erro: " + m, 8000))
        r = self.ident.report
        acc = (f" | validação: top-1 {r['file_top1']:.0%}, top-5 {r['file_top5']:.0%} por arquivo"
               if r else "")
        self.status.showMessage(f"Modelo: {len(self.ident.classes)} sinais, treinado em "
                                f"{self.ident.m.get('trained')}{acc}")
        self._load_qrg()

    # ---------- lista QRGBuscador
    def _load_qrg(self, update=False):
        if self.ident is None:
            return
        self.qrg_status.setText("lista: baixando…" if update else "lista: carregando…")

        def run():
            try:
                if update:
                    QH.update_lists()
                q = QH.QRGHints(self.ident.classes, self.ident.meta)
                self.qrg_loaded.emit(q, "")
            except Exception as e:
                self.qrg_loaded.emit(None, str(e))
        threading.Thread(target=run, daemon=True).start()

    def _update_qrg_lists(self):
        self._load_qrg(update=True)

    def _on_qrg_loaded(self, q, err):
        if err:
            self.qrg_status.setText("lista: erro")
            self.status.showMessage("Lista QRGBuscador: " + err, 10000)
            return
        self.qrg = q
        if len(q) == 0:
            self.qrg_status.setText("lista: vazia — clique em Atualizar")
            return
        n_i = sum(e["src"] == "ILGA" for e in q.entries)
        self.qrg_status.setText(f"lista: {n_i} ILGA + {len(q) - n_i} programadas")
        self._refresh_qrg()

    def _refresh_qrg(self, *_):
        if self.qrg is None or not self.use_qrg.isChecked():
            self.qrg_w, self.qrg_rows = None, []
            self.qrg_table.clear()
            self.tabs.setTabText(1, "QRG (lista)")
            self._redraw_results()
            return
        boost = (0.5, 2.0, 4.0)[self.qrg_boost.currentIndex()]
        self.qrg_w, self.qrg_rows = self.qrg.weights(self.freq.value() * 1e6, self.qrg_tol.value(),
                                                     boost=boost, use_time=self.qrg_time.isChecked())
        self.tabs.setTabText(1, f"QRG (lista) · {len(self.qrg_rows)}")
        self._fill_qrg_table()
        self._redraw_results()

    def _fill_qrg_table(self):
        self.qrg_table.clear()
        best = self.best.text()
        for r in self.qrg_rows[:300]:
            e = r["e"]
            utc = f"{e['t0']:04d}-{e['t1']:04d}" if e["t0"] is not None else ""
            ag = {True: "sim", False: "não", None: "?"}[r["active"]]
            it = QtWidgets.QTreeWidgetItem([f"{e['f']:.2f}", e["station"], e["call"], QH.describe(e),
                                            e["emis"] or e["mode"], utc, ag, e["src"],
                                            "✓" if best in r["strong"] else ""])
            tip = ("Classes do modelo reforçadas: " + ", ".join(r["classes"][:6])) if r["classes"] \
                else "Nenhuma classe do modelo associada a esta entrada"
            if e["loc"]:
                tip = f"{e['loc']}\n{tip}"
            for c in range(9):
                it.setToolTip(c, tip)
            if r["active"] is False:
                for c in range(9):
                    it.setForeground(c, QtGui.QBrush(QtGui.QColor("#808080")))
            self.qrg_table.addTopLevelItem(it)

    def _qrg_coincide(self, label):
        """Entradas da lista perto da QRG que combinam com o palpite (só informação)."""
        return [r for r in self.qrg_rows if label in r["strong"] and r["active"] is not False]

    def _fill_devices(self):
        self.dev.clear()
        try:
            apis = sd.query_hostapis()
            default_in = sd.default.device[0]
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    self.dev.addItem(f"{d['name']}  [{apis[d['hostapi']]['name']}, "
                                     f"{int(d['default_samplerate'])} Hz]", i)
                    if i == default_in:
                        self.dev.setCurrentIndex(self.dev.count() - 1)
        except Exception as e:
            self.status.showMessage(f"Erro listando áudio: {e}")
        # preferir cabo virtual, se existir
        for k in range(self.dev.count()):
            if "cable output" in self.dev.itemText(k).lower():
                self.dev.setCurrentIndex(k)
                break

    # ---------- escuta
    def _toggle_listen(self, on):
        if on:
            dev = self.dev.currentData()
            if dev is None:
                self.btn_listen.setChecked(False)
                return
            sr = int(sd.query_devices(dev)["default_samplerate"])
            try:
                self.audio.start(dev, sr)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Áudio", f"Não consegui abrir o dispositivo:\n{e}")
                self.btn_listen.setChecked(False)
                return
            self.ring = np.zeros(0, np.float32)
            nfft = 1 << int(np.ceil(np.log2(sr * 0.064)))
            self.nfft = nfft
            self.win = np.hanning(nfft).astype(np.float32)
            self.freqs = np.fft.rfftfreq(nfft, 1 / sr)
            self.nbin = int(np.searchsorted(self.freqs, min(FMAX, sr / 2)))
            self.wf = np.full((WF_ROWS, self.nbin), -120.0, np.float32)
            self.wf_img.setRect(QtCore.QRectF(0, 0, self.freqs[self.nbin - 1], WF_ROWS))
            self.btn_listen.setText("■  Parar")
            self.best_sub.setText(f"Ouvindo a {sr} Hz… aguardando {sf_.WIN_S:.0f} s de áudio")
            self.timer.start()
        else:
            self.timer.stop()
            self.audio.stop()
            self.btn_listen.setText("▶  Ouvir")

    def _tick(self):
        chunks = []
        while True:
            try:
                chunks.append(self.audio.q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return
        sr = self.audio.sr
        self.ring = np.concatenate([self.ring, *chunks])[-int(sr * 30):]
        if len(self.ring) >= self.nfft:
            seg = self.ring[-self.nfft:]
            spec = np.abs(np.fft.rfft(seg * self.win)) / (self.nfft / 4)
            db = 20 * np.log10(spec[:self.nbin] + 1e-9)
            self.wf = np.roll(self.wf, 1, axis=0)
            self.wf[0] = db
            lo, hi = np.percentile(self.wf[:60], 20), np.percentile(self.wf[:60], 99.7)
            self.wf_img.setImage(self.wf, levels=(lo, max(hi, lo + 20)), autoLevels=False)
            self.spec_curve.setData(self.freqs[:self.nbin], db)
            rms = 20 * np.log10(np.sqrt(np.mean(self.ring[-int(sr * 0.2):] ** 2)) + 1e-9)
            self.level.setValue(int(max(-90, min(0, rms))))
        self.tick_n += 1
        if self.tick_n % 20 == 0 and self.worker is not None and not self.hold:  # 1x por segundo
            n = int(sf_.WIN_S * sr)
            if len(self.ring) >= n:
                x = self.ring[-n:]
                rms = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
                if rms < SILENCE_DBFS:
                    self.best.setText("Sem áudio")
                    self.best_sub.setText(f"Nível {rms:.0f} dBFS — verifique a entrada/volume do SDR")
                    return
                self.worker.submit(sf_.to_mono_fs(x, sr))

    # ---------- resultados
    def _weights(self):
        if self.priors is None:
            return None
        f = self.freq.value() * 1e6 if self.use_freq.isChecked() else None
        m = self.mode.currentText()
        w = self.priors.weights(f, None if m.startswith("(") else m)
        if self.qrg_w is not None and self.use_qrg.isChecked():
            w = w * self.qrg_w
        return w

    def _on_result(self, p, pm, dt):
        if self.hold:
            return
        a = 1.0 / self.smooth.value() ** 0.8
        self.p_s = p if self.p_s is None else a * p + (1 - a) * self.p_s
        self.p_m = pm if self.p_m is None else a * pm + (1 - a) * self.p_m
        self._redraw_results(source=f"ao vivo · inferência {dt * 1000:.0f} ms")

    def _redraw_results(self, *_, source=None):
        if self.p_s is None or self.ident is None:
            return
        p = Priors.apply(self.p_s, self._weights())
        top = self.ident.top(p, 8)
        lab, conf = top[0]
        margin = conf - top[1][1]
        level = "alta" if conf > 0.45 and margin > 0.2 else "média" if conf > 0.22 else "baixa"
        changed = lab != self.best.text()
        self.best.setText(lab)
        info = self.ident.info(lab).get("fields", {})
        fr = info.get("frequencies", "")
        fr = fr if len(fr) <= 70 else fr[:67] + "…"
        coinc = self._qrg_coincide(lab) if self.use_qrg.isChecked() else []
        qrg_txt = ""
        if coinc:
            e = coinc[0]["e"]
            more = f" (+{len(coinc) - 1})" if len(coinc) > 1 else ""
            qrg_txt = (f"\nNa lista: {e['f']:.2f} kHz · {e['station']}"
                       + (f" · indicativo {e['call']}" if e["call"] else "") + f" · {QH.describe(e)[:60]}{more}")
        self.best_sub.setText(f"Confiança {conf:.0%} ({level})"
                              + (f" · {source}" if source else "")
                              + (f"\n{fr}  ·  {info.get('mode', '')}  ·  "
                                 f"{info.get('modulation', '')}  ·  {info.get('bandwidth', '')}" if info else "")
                              + qrg_txt)
        if changed and self.qrg_rows:
            self._fill_qrg_table()
        mods = ", ".join(f"{m} {v:.0%}" for m, v in self.ident.top_mod(self.p_m))
        self.mod_label.setText(f"<b>Modulação provável:</b> {mods}")
        sel = self.table.currentItem().text(1) if self.table.currentItem() else None
        self.table.clear()
        for i, (l, v) in enumerate(top, 1):
            f = self.ident.info(l).get("fields", {})
            it = QtWidgets.QTreeWidgetItem([str(i), l, f"{'█' * int(round(v * 20)):<20} {v:5.1%}",
                                            f.get("mode", ""), f.get("modulation", "")])
            it.setToolTip(1, l)
            self.table.addTopLevelItem(it)
            if l == (sel or lab):
                self.table.setCurrentItem(it)
        if level != "baixa" and lab != self.last_best:
            self.last_best = lab
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            fr = (f" @ {self.freq.value():.4f} MHz"
                  if self.use_freq.isChecked() or self.use_qrg.isChecked() else "")
            extra = ""
            if coinc:
                e = coinc[0]["e"]
                extra = f"  · lista: {e['station']}" + (f" {e['call']}" if e["call"] else "")
            self.history.insertItem(0, f"{ts}{fr}  {lab}  ({conf:.0%}){extra}")

    def _show_details(self):
        it = self.table.currentItem()
        if not it or self.ident is None:
            return
        lab = it.text(1)
        m = self.ident.info(lab)
        f = m.get("fields", {})
        rows = "".join(f"<tr><td><b>{k.title()}</b></td><td>{v}</td></tr>"
                       for k, v in f.items() if k in ("frequencies", "mode", "modulation", "bandwidth",
                                                     "location", "acf", "emission designator"))
        cats = ", ".join(m.get("categories", []))
        url = m.get("url", "")
        self.details.setHtml(f"<h3>{lab}</h3><table cellpadding=2>{rows}</table>"
                             f"<p>{m.get('description', '')}</p>"
                             f"<p><i>{cats}</i></p>"
                             + (f'<p><a href="{url}">{url}</a></p>' if url else ""))

    def _open_wiki(self):
        it = self.table.currentItem()
        if it and self.ident:
            url = self.ident.info(it.text(1)).get("url")
            if url:
                webbrowser.open(url)

    def _reset(self):
        self.p_s = self.p_m = None
        self.last_best = None
        self.table.clear()
        self.best.setText("—")
        self.best_sub.setText("")
        self.mod_label.setText("")

    # ---------- arquivos
    def _open_file(self):
        if self.worker is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Arquivo de áudio", "", "Áudio (*.wav *.mp3 *.flac *.ogg);;Todos (*.*)")
        if path:
            self.best.setText("Analisando…")
            self.best_sub.setText(os.path.basename(path))
            self.worker.analyze_file(path)

    def _on_file_result(self, path, p, pm, P):
        self.p_s, self.p_m = p, pm
        self.last_best = None
        self._redraw_results(source=f"arquivo {os.path.basename(path)} ({len(P)} janelas)")

    def _save_recording(self):
        if len(self.ring) == 0:
            return
        os.makedirs(REC_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        lab = self.best.text().replace("/", "-")[:40] if self.p_s is not None else "desconhecido"
        lab = "".join(c for c in lab if c.isalnum() or c in " -_().").strip()
        fr = f"_{self.freq.value():.4f}MHz" if self.use_freq.isChecked() else ""
        path = os.path.join(REC_DIR, f"{ts}{fr}_{lab}.wav")
        sf.write(path, self.ring, self.audio.sr)
        self.status.showMessage(f"Gravado: {path}", 8000)

    def closeEvent(self, e):
        self.timer.stop()
        self.audio.stop()
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, c in [(QtGui.QPalette.ColorRole.Window, "#1e1f24"), (QtGui.QPalette.ColorRole.WindowText, "#e6e6e6"),
                    (QtGui.QPalette.ColorRole.Base, "#15161a"), (QtGui.QPalette.ColorRole.AlternateBase, "#24262c"),
                    (QtGui.QPalette.ColorRole.Text, "#e6e6e6"), (QtGui.QPalette.ColorRole.Button, "#2b2d34"),
                    (QtGui.QPalette.ColorRole.ButtonText, "#e6e6e6"), (QtGui.QPalette.ColorRole.Highlight, "#1976d2"),
                    (QtGui.QPalette.ColorRole.HighlightedText, "#ffffff"), (QtGui.QPalette.ColorRole.Link, "#64b5f6")]:
        pal.setColor(role, QtGui.QColor(c))
    app.setPalette(pal)
    w = Main()
    w.setWindowTitle(f"SigID Shazam {VERSION} — identificação de sinais de rádio")
    w.show()
    selftest = os.environ.get("SIGID_SELFTEST")
    if selftest:  # autoteste do executável: carrega modelo/lista, abre o áudio e grava um relatório
        def report():
            try:
                lines = [f"modelo={'ok' if w.ident else 'FALHOU'} classes={len(w.ident.classes) if w.ident else 0}",
                         f"lista={w.qrg_status.text()}",
                         f"dispositivos={w.dev.count()}"]
                if w.ident:
                    import numpy as np
                    t = np.arange(64000) / 16000
                    p, pm = w.ident.proba((np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(64000)).astype(np.float32))
                    lines.append(f"inferencia=ok top={w.ident.top(p, 1)[0][0]} mod={w.ident.top_mod(pm, 1)[0][0]}")
            except Exception as e:
                lines = [f"ERRO {e!r}"]
            with open(selftest, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            app.quit()
        QtCore.QTimer.singleShot(8000, report)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
