package br.pp2la.sigid.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import br.pp2la.sigid.model.Engine
import br.pp2la.sigid.model.QrgRow
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.materialswitch.MaterialSwitch
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.pow
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private var engine: Engine? = null
    private lateinit var audio: AudioEngine
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)

    private var pS: DoubleArray? = null
    private var pM: DoubleArray? = null
    private var hold = false
    private var lastBest: String? = null
    private var qrgW: DoubleArray? = null
    private var qrgRows: List<QrgRow> = emptyList()
    private val history = ArrayList<String>()

    // views
    private lateinit var wf: WaterfallView
    private lateinit var btnListen: MaterialButton
    private lateinit var level: ProgressBar
    private lateinit var levelTxt: TextView
    private lateinit var best: TextView
    private lateinit var bestSub: TextView
    private lateinit var modTxt: TextView
    private lateinit var topList: LinearLayout
    private lateinit var status: TextView
    private lateinit var histTxt: TextView
    private lateinit var freqEdit: EditText
    private lateinit var swRange: MaterialSwitch
    private lateinit var modeSpin: Spinner
    private lateinit var swQrg: MaterialSwitch
    private lateinit var tolEdit: EditText
    private lateinit var weightSpin: Spinner
    private lateinit var swTime: MaterialSwitch
    private lateinit var btnQrg: MaterialButton
    private lateinit var btnHold: MaterialButton
    private lateinit var smooth: SeekBar
    private val prefs by lazy { getSharedPreferences("sigid", MODE_PRIVATE) }

    private val askMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startListening() else toast("Sem permissão de microfone não dá para ouvir.")
    }
    private val pickFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) analyzeFile(uri)
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).roundToInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        audio = AudioEngine(this)
        buildUi()
        status.text = "Carregando modelo…"
        worker.execute {
            try {
                val e = Engine(this)
                main.post {
                    engine = e
                    val r = e.report
                    val acc = if (r.has("file_top1")) " · validação (via microfone simulado): top-1 " +
                            "${(r.getDouble("file_top1") * 100).roundToInt()}%, top-5 ${(r.getDouble("file_top5") * 100).roundToInt()}%" else ""
                    status.text = "Modelo: ${e.classes.size} sinais$acc"
                }
                e.loadQrg(this)
                main.post {
                    val n = e.qrg?.entries?.size ?: 0
                    status.append("\nLista QRGBuscador: $n entradas")
                    refreshQrg()
                }
            } catch (t: Throwable) {
                main.post { status.text = "Erro ao carregar o modelo: ${t.message}" }
            }
        }
    }

    // ------------------------------------------------------------------ interface
    private fun card(parent: LinearLayout, title: String? = null): LinearLayout {
        val c = MaterialCardView(this).apply {
            radius = dp(14).toFloat(); setCardBackgroundColor(Color.parseColor("#22242A")); cardElevation = 0f
            strokeWidth = 0
        }
        val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        lp.setMargins(dp(10), dp(6), dp(10), dp(6))
        parent.addView(c, lp)
        val inner = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(14), dp(12), dp(14), dp(12)) }
        c.addView(inner)
        if (title != null) inner.addView(text(title, 13f, "#8AB4F8", bold = true))
        return inner
    }

    private fun text(s: String, size: Float = 14f, color: String = "#E6E6E6", bold: Boolean = false) =
        TextView(this).apply {
            text = s; textSize = size; setTextColor(Color.parseColor(color))
            if (bold) setTypeface(typeface, Typeface.BOLD)
        }

    private fun row(parent: LinearLayout): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        parent.addView(this, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    }

    private fun button(s: String, outline: Boolean = true, onClick: (MaterialButton) -> Unit): MaterialButton {
        val b = if (outline) MaterialButton(this, null, com.google.android.material.R.attr.materialButtonOutlinedStyle)
        else MaterialButton(this)
        b.text = s; b.isAllCaps = false; b.setOnClickListener { onClick(b) }
        return b
    }

    private fun spinner(items: List<String>, sel: Int): Spinner = Spinner(this).apply {
        adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, items)
        setSelection(sel)
    }

    private fun buildUi() {
        val scroll = ScrollView(this)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(8), 0, dp(24)) }
        scroll.addView(root)
        setContentView(scroll)

        val head = row(root).apply { setPadding(dp(16), dp(4), dp(16), dp(4)) }
        head.addView(text("SigID Shazam", 22f, "#FFFFFF", bold = true), LinearLayout.LayoutParams(0, -2, 1f))
        btnListen = button("▶  Ouvir", outline = false) { toggleListen() }
        head.addView(btnListen)

        val lv = row(root).apply { setPadding(dp(16), 0, dp(16), dp(4)) }
        level = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 90; progress = 0 }
        lv.addView(level, LinearLayout.LayoutParams(0, dp(10), 1f))
        levelTxt = text("  — dBFS", 12f, "#AAAAAA")
        lv.addView(levelTxt)

        wf = WaterfallView(this)
        val wlp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(190))
        wlp.setMargins(dp(10), dp(4), dp(10), dp(4))
        root.addView(wf, wlp)
        audio.onBlock = { blk ->
            wf.push(blk)
            val db = Engine.rmsDb(blk)
            main.post { level.progress = (db + 90).toInt().coerceIn(0, 90); levelTxt.text = "  %.0f dBFS".format(db) }
        }

        val res = card(root)
        best = text("—", 21f, "#FFFFFF", bold = true)
        res.addView(best)
        bestSub = text("Aproxime o celular do alto-falante do rádio e toque em Ouvir.", 13f, "#BBBBBB")
        res.addView(bestSub)
        modTxt = text("", 13f, "#FFD54F")
        res.addView(modTxt)
        topList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(6), 0, 0) }
        res.addView(topList)

        val ctl = row(res)
        btnHold = button("Congelar") { hold = !hold; it.text = if (hold) "Descongelar" else "Congelar" }
        ctl.addView(btnHold, LinearLayout.LayoutParams(0, -2, 1f))
        ctl.addView(button("Limpar") { resetResults() }, LinearLayout.LayoutParams(0, -2, 1f))
        val ctl2 = row(res)
        ctl2.addView(button("● Salvar 30 s") { saveRecording() }, LinearLayout.LayoutParams(0, -2, 1f))
        ctl2.addView(button("Arquivo…") { pickFile.launch(arrayOf("audio/*")) }, LinearLayout.LayoutParams(0, -2, 1f))

        // dicas
        val hc = card(root, "DICAS (opcionais — só reforçam, o áudio decide)")
        val r1 = row(hc)
        r1.addView(text("Frequência (MHz) ", 14f))
        freqEdit = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(prefs.getString("freq", "7.1")); setTextColor(Color.WHITE); setSelectAllOnFocus(true)
        }
        r1.addView(freqEdit, LinearLayout.LayoutParams(0, -2, 1f))
        swRange = MaterialSwitch(this).apply { text = "Faixas do sigidwiki"; isChecked = prefs.getBoolean("range", false) }
        hc.addView(swRange)
        val r2 = row(hc)
        r2.addView(text("Modo do rádio  ", 14f))
        modeSpin = spinner(listOf("(não usar)", "USB", "LSB", "CW", "AM", "NFM", "WFM"), prefs.getInt("mode", 0))
        r2.addView(modeSpin, LinearLayout.LayoutParams(0, -2, 1f))
        swQrg = MaterialSwitch(this).apply { text = "Lista QRGBuscador (ILGA + programadas)"; isChecked = prefs.getBoolean("qrg", false) }
        hc.addView(swQrg)
        val r3 = row(hc)
        r3.addView(text("±  ", 14f))
        tolEdit = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(prefs.getString("tol", "2.0")); setTextColor(Color.WHITE); setSelectAllOnFocus(true)
        }
        r3.addView(tolEdit, LinearLayout.LayoutParams(dp(70), -2))
        r3.addView(text(" kHz   peso ", 14f))
        weightSpin = spinner(listOf("leve (~1,5×)", "médio (~3×)", "forte (~5×)"), prefs.getInt("weight", 1))
        r3.addView(weightSpin, LinearLayout.LayoutParams(0, -2, 1f))
        swTime = MaterialSwitch(this).apply { text = "Considerar horário UTC"; isChecked = prefs.getBoolean("utc", true) }
        hc.addView(swTime)
        btnQrg = button("Ver QRGs da lista") { showQrgDialog() }
        hc.addView(btnQrg)
        val r4 = row(hc)
        r4.addView(text("Suavização  ", 14f))
        smooth = SeekBar(this).apply { max = 9; progress = prefs.getInt("smooth", 5) }
        r4.addView(smooth, LinearLayout.LayoutParams(0, -2, 1f))

        val onHint = { saveHints(); refreshQrg() }
        freqEdit.setOnFocusChangeListener { _, f -> if (!f) onHint() }
        freqEdit.setOnEditorActionListener { _, _, _ -> onHint(); false }
        tolEdit.setOnFocusChangeListener { _, f -> if (!f) onHint() }
        tolEdit.setOnEditorActionListener { _, _, _ -> onHint(); false }
        for (s in listOf(swRange, swQrg, swTime)) s.setOnCheckedChangeListener { _, _ -> onHint() }
        val sel = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) { onHint() }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
        modeSpin.onItemSelectedListener = sel
        weightSpin.onItemSelectedListener = sel
        smooth.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, u: Boolean) { prefs.edit().putInt("smooth", p).apply() }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })

        val hcard = card(root, "HISTÓRICO")
        histTxt = text("—", 13f, "#CCCCCC")
        hcard.addView(histTxt)
        val ab = card(root)
        status = text("", 12f, "#999999")
        ab.addView(status)
        ab.addView(text("Treinado com amostras do Signal Identification Wiki (sigidwiki.com). " +
                "Lista de QRGs: github.com/lualcorreia/QRGBuscador.", 11f, "#777777"))
        main.postDelayed(object : Runnable { override fun run() { refreshQrg(); main.postDelayed(this, 60000) } }, 60000)
    }

    private fun saveHints() {
        prefs.edit().putString("freq", freqEdit.text.toString()).putBoolean("range", swRange.isChecked)
            .putInt("mode", modeSpin.selectedItemPosition).putBoolean("qrg", swQrg.isChecked)
            .putString("tol", tolEdit.text.toString()).putInt("weight", weightSpin.selectedItemPosition)
            .putBoolean("utc", swTime.isChecked).apply()
    }

    private fun freqHz(): Double? = freqEdit.text.toString().replace(",", ".").toDoubleOrNull()?.let { it * 1e6 }

    // ------------------------------------------------------------------ escuta
    private fun toggleListen() {
        if (audio.isRunning) stopListening()
        else if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
            startListening()
        else askMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun startListening() {
        if (!audio.start(android.media.MediaRecorder.AudioSource.UNPROCESSED)) { toast("Não consegui abrir o microfone."); return }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        btnListen.text = "■  Parar"
        bestSub.text = "Ouvindo pelo ${audio.sourceName}… aguardando 4 s de áudio"
        main.postDelayed(tick, 1000)
    }

    private fun stopListening() {
        main.removeCallbacks(tick)
        audio.stop()
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        btnListen.text = "▶  Ouvir"
    }

    private val tick = object : Runnable {
        override fun run() {
            if (!audio.isRunning) return
            main.postDelayed(this, 1000)
            val e = engine ?: return
            if (hold || audio.available() < 4.0 || !busy.compareAndSet(false, true)) return
            val x = audio.last(64000)
            val db = Engine.rmsDb(x)
            if (db < -72) {
                busy.set(false)
                best.text = "Sem áudio"; bestSub.text = "Nível %.0f dBFS — aumente o volume do rádio".format(db)
                return
            }
            worker.execute {
                try {
                    val r = e.identify(x)
                    main.post { onResult(r.p, r.pm, "ao vivo · ${r.ms} ms") }
                } catch (t: Throwable) {
                    main.post { status.text = "Erro: ${t.message}" }
                } finally { busy.set(false) }
            }
        }
    }

    private fun onResult(p: DoubleArray, pm: DoubleArray, source: String) {
        if (hold) return
        val a = 1.0 / (smooth.progress + 1).toDouble().pow(0.8)
        pS = pS?.let { old -> DoubleArray(p.size) { a * p[it] + (1 - a) * old[it] } } ?: p
        pM = pM?.let { old -> DoubleArray(pm.size) { a * pm[it] + (1 - a) * old[it] } } ?: pm
        redraw(source)
    }

    // ------------------------------------------------------------------ resultados
    private fun weights(): DoubleArray? {
        val e = engine ?: return null
        val m = modeSpin.selectedItem.toString()
        var w = e.priors.weights(if (swRange.isChecked) freqHz() else null, if (m.startsWith("(")) null else m)
        qrgW?.let { q -> if (swQrg.isChecked) w = DoubleArray(w.size) { w[it] * q[it] } }
        return w
    }

    private fun refreshQrg() {
        val e = engine; val q = e?.qrg; val f = freqHz()
        if (q == null || !swQrg.isChecked || f == null) {
            qrgW = null; qrgRows = emptyList(); btnQrg.text = "Ver QRGs da lista"; redraw(null); return
        }
        val tol = tolEdit.text.toString().replace(",", ".").toDoubleOrNull() ?: 2.0
        val boost = doubleArrayOf(0.5, 2.0, 4.0)[weightSpin.selectedItemPosition]
        val (w, rows) = q.weights(f, tol, boost, swTime.isChecked)
        qrgW = w; qrgRows = rows
        btnQrg.text = "Ver QRGs da lista (${rows.size})"
        redraw(null)
    }

    private var lastSource: String? = null

    private fun redraw(source: String?) {
        val e = engine ?: return
        val ps = pS ?: return
        val pm = pM ?: return
        if (source != null) lastSource = source
        val w = weights() ?: return
        val q = DoubleArray(ps.size) { ps[it] * w[it] }
        val s = q.sum()
        for (i in q.indices) q[i] /= s
        val order = q.indices.sortedByDescending { q[it] }.take(8)
        val bi = order[0]; val conf = q[bi]; val margin = conf - q[order[1]]
        val lvl = if (conf > 0.45 && margin > 0.2) "alta" else if (conf > 0.22) "média" else "baixa"
        val info = e.signals[bi]
        best.text = info.name
        val fr = info.fields["frequencies"].orEmpty().let { if (it.length > 60) it.take(57) + "…" else it }
        val coinc = if (swQrg.isChecked) qrgRows.filter { bi in it.strong && it.active != false } else emptyList()
        val sb = StringBuilder("Confiança ${(conf * 100).roundToInt()}% ($lvl)")
        lastSource?.let { sb.append(" · ").append(it) }
        if (info.fields.isNotEmpty()) sb.append("\n").append(listOf(fr, info.fields["mode"], info.fields["modulation"],
            info.fields["bandwidth"]).filter { !it.isNullOrBlank() }.joinToString("  ·  "))
        if (coinc.isNotEmpty()) {
            val c = coinc[0].e
            sb.append("\nNa lista: %.2f kHz · %s".format(c.f, c.station))
            if (c.call.isNotBlank()) sb.append(" · indicativo ").append(c.call)
            sb.append(" · ").append(c.desc.take(50))
            if (coinc.size > 1) sb.append(" (+${coinc.size - 1})")
        }
        bestSub.text = sb.toString()
        val mo = pm.indices.sortedByDescending { pm[it] }.take(3)
        modTxt.text = "Modulação provável: " + mo.joinToString(", ") { "${e.modClasses[it]} ${(pm[it] * 100).roundToInt()}%" }
        topList.removeAllViews()
        for ((rank, i) in order.withIndex()) {
            val pct = q[i]
            val bar = "█".repeat((pct * 20).roundToInt().coerceIn(0, 20))
            val tv = text("${rank + 1}. ${e.signals[i].name}\n    $bar ${"%.1f".format(pct * 100)}%   " +
                    listOf(e.signals[i].fields["mode"], e.signals[i].fields["modulation"]).filter { !it.isNullOrBlank() }
                        .joinToString(" · "), 13f, if (rank == 0) "#FFFFFF" else "#CFCFCF")
            tv.setPadding(0, dp(4), 0, dp(4))
            tv.setOnClickListener { showDetails(i) }
            topList.addView(tv)
        }
        if (lvl != "baixa" && info.name != lastBest) {
            lastBest = info.name
            val ts = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
            val f = if (swRange.isChecked || swQrg.isChecked) freqHz()?.let { " @ %.4f MHz".format(it / 1e6) } ?: "" else ""
            val extra = coinc.firstOrNull()?.e?.let { "  · lista: ${it.station} ${it.call}".trimEnd() } ?: ""
            history.add(0, "$ts$f  ${info.name}  (${(conf * 100).roundToInt()}%)$extra")
            while (history.size > 40) history.removeAt(history.size - 1)
            histTxt.text = history.joinToString("\n")
        }
    }

    private fun resetResults() {
        pS = null; pM = null; lastBest = null; lastSource = null
        best.text = "—"; bestSub.text = ""; modTxt.text = ""; topList.removeAllViews()
    }

    private fun showDetails(i: Int) {
        val e = engine ?: return
        val s = e.signals[i]
        val msg = buildString {
            for ((k, v) in s.fields) append(k.replaceFirstChar { it.uppercase() }).append(": ").append(v).append("\n")
            if (s.description.isNotBlank()) append("\n").append(s.description).append("\n")
            if (s.categories.isNotEmpty()) append("\n").append(s.categories.joinToString(", "))
        }
        val b = AlertDialog.Builder(this).setTitle(s.name).setMessage(msg).setNegativeButton("Fechar", null)
        if (s.url.isNotBlank()) b.setPositiveButton("Abrir no sigidwiki") { _, _ ->
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(s.url)))
        }
        b.show()
    }

    private fun showQrgDialog() {
        val e = engine ?: return
        if (qrgRows.isEmpty()) { toast("Ligue a lista QRGBuscador e digite a frequência."); return }
        val bi = pS?.let { ps -> weights()?.let { w -> ps.indices.maxBy { ps[it] * w[it] } } }
        val lines = qrgRows.take(300).map { r ->
            val x = r.e
            val utc = if (x.t0 != null) " %04d-%04d".format(x.t0, x.t1) else ""
            val now = when (r.active) { true -> " · agora"; false -> " · fora do horário"; null -> "" }
            val ok = if (bi != null && bi in r.strong) "✓ " else ""
            val khz = String.format(Locale.US, "%.2f", x.f)
            "$ok$khz kHz  ${x.station}${if (x.call.isNotBlank()) "  [${x.call}]" else ""}\n   ${x.desc}  ${x.emis}$utc$now · ${x.src}" +
                (if (r.classes.isNotEmpty()) "\n   → " + r.classes.take(3).joinToString(", ") { e.classes[it] } else "")
        }
        AlertDialog.Builder(this).setTitle("QRGs próximas (${qrgRows.size})")
            .setItems(lines.toTypedArray(), null).setNegativeButton("Fechar", null).show()
    }

    // ------------------------------------------------------------------ arquivos
    private fun saveRecording() {
        val x = audio.last(AudioEngine.FS * 30)
        if (x.isEmpty()) { toast("Nada gravado ainda — toque em Ouvir primeiro."); return }
        val dir = File(getExternalFilesDir(null), "gravacoes").apply { mkdirs() }
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val lab = best.text.toString().filter { it.isLetterOrDigit() || it in " -_()." }.take(40).trim()
        val f = freqHz()?.let { "_%.4fMHz".format(it / 1e6) } ?: ""
        val file = File(dir, "${ts}${f}_$lab.wav")
        writeWav(file, x, AudioEngine.FS)
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "audio/wav"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "Gravado: ${file.name}"))
    }

    private fun writeWav(file: File, x: FloatArray, sr: Int) {
        val data = ByteBuffer.allocate(x.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in x) data.putShort((v.coerceIn(-1f, 1f) * 32767).toInt().toShort())
        val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()); h.putInt(36 + x.size * 2); h.put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()); h.putInt(16); h.putShort(1); h.putShort(1); h.putInt(sr); h.putInt(sr * 2)
        h.putShort(2); h.putShort(16); h.put("data".toByteArray()); h.putInt(x.size * 2)
        FileOutputStream(file).use { it.write(h.array()); it.write(data.array()) }
    }

    private fun analyzeFile(uri: Uri) {
        val e = engine ?: return
        best.text = "Analisando…"; bestSub.text = uri.lastPathSegment ?: ""
        worker.execute {
            try {
                val x = FileDecoder.decodeTo16k(this, uri)
                val r = e.identifyLong(x) { i, n -> main.post { bestSub.text = "Analisando janela ${i + 1}/$n…" } }
                main.post { pS = r.p; pM = r.pm; lastBest = null; redraw("arquivo (${"%.0f".format(x.size / 16000.0)} s)") }
            } catch (t: Throwable) {
                main.post { best.text = "Erro no arquivo"; bestSub.text = t.message ?: t.toString() }
            }
        }
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_LONG).show()

    override fun onPause() { super.onPause(); saveHints() }

    override fun onDestroy() {
        super.onDestroy()
        audio.stop()
        worker.shutdownNow()
    }
}
