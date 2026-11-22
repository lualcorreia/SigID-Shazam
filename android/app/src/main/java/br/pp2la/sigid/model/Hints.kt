package br.pp2la.sigid.model

import java.io.BufferedReader
import java.util.Calendar
import java.util.TimeZone

/** Pesos a partir da frequência sintonizada (faixas do sigidwiki) e do modo de demodulação. */
class Priors(private val ranges: List<List<DoubleArray>>, private val fams: List<Set<String>>) {
    companion object {
        val MODE_FAMILY = mapOf("USB" to "SSB", "LSB" to "SSB", "SSB" to "SSB", "CW" to "SSB",
            "NFM" to "FM", "FM" to "FM", "WFM" to "FM", "AM" to "AM")
    }

    fun weights(freqHz: Double?, mode: String?): DoubleArray {
        val w = DoubleArray(ranges.size) { 1.0 }
        if (freqHz != null) for (i in ranges.indices) {
            val rr = ranges[i]
            if (rr.isNotEmpty() && rr.none { freqHz >= it[0] && freqHz <= it[1] }) w[i] *= 0.2
        }
        val fam = MODE_FAMILY[mode?.uppercase()]
        if (fam != null) for (i in fams.indices) if (fams[i].isNotEmpty() && fam !in fams[i]) w[i] *= 0.4
        return w
    }

    fun inRange(i: Int, freqHz: Double) = ranges[i].any { freqHz >= it[0] && freqHz <= it[1] }
}

class QrgEntry(val f: Double, val station: String, val call: String, val desc: String, val emis: String,
               val t0: Int?, val t1: Int?, val days: String, val src: String, val loc: String, val prof: Int)

class QrgProfile(val cls: IntArray, val str: DoubleArray, val fam: Map<String, Double>)

class QrgRow(val e: QrgEntry, val d: Double, val active: Boolean?, val classes: List<Int>, val strong: List<Int>)

/** Lista QRGBuscador (ILGA + programadas): só um empurrão (peso >= 1); quem decide é o áudio. */
class QrgHints(val entries: List<QrgEntry>, private val profiles: List<QrgProfile>,
               private val classFams: List<String>, private val priors: Priors, private val nClasses: Int) {
    private val freqs = DoubleArray(entries.size) { entries[it].f }

    companion object {
        fun load(tsv: BufferedReader, prof: BufferedReader, classFams: List<String>, priors: Priors, n: Int): QrgHints {
            val profiles = prof.readLines().map { line ->
                val parts = line.split("\t")
                val cl = parts.getOrElse(0) { "" }.split(",").filter { it.isNotBlank() }
                val fa = parts.getOrElse(1) { "" }.split(",").filter { it.isNotBlank() }
                QrgProfile(IntArray(cl.size) { cl[it].substringBefore(":").toInt() },
                    DoubleArray(cl.size) { cl[it].substringAfter(":").toDouble() },
                    fa.associate { it.substringBefore(":") to it.substringAfter(":").toDouble() })
            }
            val entries = ArrayList<QrgEntry>(80000)
            tsv.forEachLine { line ->
                val c = line.split("\t")
                if (c.size >= 11) entries.add(QrgEntry(c[0].toDouble(), c[1], c[2], c[3], c[4], c[5].toIntOrNull(),
                    c[6].toIntOrNull(), c[7], c[8], c[9], c[10].toInt()))
            }
            entries.sortBy { it.f }
            return QrgHints(entries, profiles, classFams, priors, n)
        }
    }

    fun near(fKhz: Double, tol: Double): List<QrgEntry> {
        var a = lower(fKhz - tol)
        val out = ArrayList<QrgEntry>()
        while (a < freqs.size && freqs[a] <= fKhz + tol) { out.add(entries[a]); a++ }
        return out
    }

    private fun lower(v: Double): Int {
        var lo = 0; var hi = freqs.size
        while (lo < hi) { val m = (lo + hi) ushr 1; if (freqs[m] < v) lo = m + 1 else hi = m }
        return lo
    }

    fun isActive(e: QrgEntry, now: Calendar): Boolean? {
        val t0 = e.t0 ?: return null
        val t1 = e.t1 ?: return null
        val hm = now.get(Calendar.HOUR_OF_DAY) * 100 + now.get(Calendar.MINUTE)
        var on = when {
            t0 == t1 || (t0 == 0 && t1 >= 2400) -> true
            t0 < t1 -> hm in t0 until t1
            else -> hm >= t0 || hm < t1
        }
        val d = now.get(Calendar.DAY_OF_WEEK).toString() // 1 = domingo, como na ILGA
        if (e.days.isNotEmpty() && !e.days.contains(d)) on = false
        return on
    }

    fun weights(freqHz: Double, tolKhz: Double, boost: Double, useTime: Boolean): Pair<DoubleArray, List<QrgRow>> {
        val w = DoubleArray(nClasses) { 1.0 }
        val now = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
        val f = freqHz / 1e3
        val score = DoubleArray(nClasses)
        val famScore = HashMap<String, Double>()
        val rows = ArrayList<QrgRow>()
        for (e in near(f, tolKhz)) {
            val act = if (useTime) isActive(e, now) else null
            val tw = when (act) { null -> 0.7; true -> 1.0; false -> 0.35 }
            val d = kotlin.math.abs(e.f - f)
            val dw = 1.0 - 0.5 * d / kotlin.math.max(tolKhz, 1e-6)
            val pr = profiles[e.prof]
            for (j in pr.cls.indices) {
                val v = pr.str[j] * tw * dw
                if (v > score[pr.cls[j]]) score[pr.cls[j]] = v
            }
            for ((k, s) in pr.fam) famScore[k] = kotlin.math.max(famScore[k] ?: 0.0, s * tw * dw)
            val order = pr.cls.indices.sortedByDescending { pr.str[it] }
            rows.add(QrgRow(e, d, act, order.map { pr.cls[it] }, order.filter { pr.str[it] >= 0.6 }.map { pr.cls[it] }))
        }
        for (i in 0 until nClasses) w[i] += boost * score[i]
        for (i in 0 until nClasses) {
            val fam = classFams[i]
            val fs = famScore[fam]
            if (fam.isNotEmpty() && fs != null && priors.inRange(i, freqHz)) w[i] *= 1 + 0.25 * fs
        }
        rows.sortWith(compareBy<QrgRow> { it.active == false }.thenBy { it.d })
        return w to rows
    }
}
