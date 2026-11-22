package br.pp2la.sigid.model

import android.content.Context
import org.json.JSONObject
import java.util.zip.GZIPInputStream
import kotlin.math.log10
import kotlin.math.sqrt

class SignalInfo(val name: String, val fields: Map<String, String>, val description: String, val url: String,
                 val categories: List<String>)

/** Carrega modelo, metadados e listas QRG dos assets; identifica janelas de 4 s a 16 kHz. */
class Engine(ctx: Context) {
    val meta: JSONObject = JSONObject(ctx.assets.open("meta.json").bufferedReader().readText())
    val classes: List<String>
    val modClasses: List<String>
    val signals: List<SignalInfo>
    val model: SigModel
    val priors: Priors
    val report: JSONObject = meta.optJSONObject("report") ?: JSONObject()
    val noiseIndex: Int
    @Volatile var qrg: QrgHints? = null
        private set
    private val classFams: List<String>

    init {
        val cl = meta.getJSONArray("classes")
        classes = List(cl.length()) { cl.getString(it) }
        val mc = meta.getJSONArray("mod_classes")
        modClasses = List(mc.length()) { mc.getString(it) }
        val sg = meta.getJSONArray("signals")
        val ranges = ArrayList<List<DoubleArray>>()
        val fams = ArrayList<Set<String>>()
        signals = List(sg.length()) { i ->
            val o = sg.getJSONObject(i)
            val fo = o.optJSONObject("fields") ?: JSONObject()
            val fields = fo.keys().asSequence().associateWith { fo.getString(it) }
            val rr = o.optJSONArray("ranges")
            ranges.add(List(rr?.length() ?: 0) { j -> rr!!.getJSONArray(j).let { doubleArrayOf(it.getDouble(0), it.getDouble(1)) } })
            val fa = o.optJSONArray("fams")
            fams.add((0 until (fa?.length() ?: 0)).map { fa!!.getString(it) }.toSet())
            val ca = o.optJSONArray("categories")
            SignalInfo(classes[i], fields, o.optString("description"), o.optString("url"),
                List(ca?.length() ?: 0) { ca!!.getString(it) })
        }
        priors = Priors(ranges, fams)
        classFams = ctx.assets.open("class_fams.txt").bufferedReader().readLines()
            .let { l -> List(classes.size) { l.getOrElse(it) { "" } } }
        val arrays = ctx.assets.open("model.bin").use { BinArrays.read(it) }
        model = SigModel(arrays, meta.getInt("n_mlp"), meta.getInt("n_layers"), meta.getInt("n_mod_layers"),
            classes.size, meta.optDouble("T", 2.0), meta.optDouble("w_mlp", 0.65), meta.optInt("knn_k", 7))
        noiseIndex = classes.indexOfFirst { it.startsWith("Ruído") }
    }

    fun loadQrg(ctx: Context) {
        val tsv = GZIPInputStream(ctx.assets.open("qrg.tsv.gz")).bufferedReader()
        val prof = ctx.assets.open("qrg_profiles.tsv").bufferedReader()
        qrg = QrgHints.load(tsv, prof, classFams, priors, classes.size)
    }

    class Result(val p: DoubleArray, val pm: DoubleArray, val ms: Long)

    fun identify(x16: FloatArray): Result {
        val t = System.currentTimeMillis()
        val f = model.features.extract(x16)
        val (p, pm) = model.proba(f)
        return Result(p, pm, System.currentTimeMillis() - t)
    }

    /** Equivalente a sigfeatures.segments(hop 2 s, max 60, keep_db 20) + média das janelas. */
    fun identifyLong(x16: FloatArray, progress: (Int, Int) -> Unit): Result {
        val n = SigModel.WIN
        val segs = ArrayList<FloatArray>()
        if (x16.size <= n) segs.add(x16) else {
            val hop = 2 * 16000
            val all = (0..x16.size - n step hop).map { s -> x16.copyOfRange(s, s + n) }
            val rms = all.map { s -> var a = 0.0; for (v in s) a += v.toDouble() * v; 10 * log10(a / s.size + 1e-12) }
            val mx = rms.max()
            val keep = all.filterIndexed { i, _ -> rms[i] >= mx - 20 }
            if (keep.size > 60) for (i in 0 until 60) segs.add(keep[Math.round(i * (keep.size - 1) / 59.0).toInt()])
            else segs.addAll(keep)
        }
        val t = System.currentTimeMillis()
        val p = DoubleArray(classes.size); val pm = DoubleArray(modClasses.size)
        for ((i, s) in segs.withIndex()) {
            progress(i, segs.size)
            val r = identify(s)
            for (j in p.indices) p[j] += r.p[j] / segs.size
            for (j in pm.indices) pm[j] += r.pm[j] / segs.size
        }
        return Result(p, pm, System.currentTimeMillis() - t)
    }

    companion object {
        fun rmsDb(x: FloatArray, from: Int = 0, to: Int = x.size): Double {
            var a = 0.0
            for (i in from until to) a += x[i].toDouble() * x[i]
            return 20 * log10(sqrt(a / maxOf(1, to - from)) + 1e-12)
        }
    }
}
