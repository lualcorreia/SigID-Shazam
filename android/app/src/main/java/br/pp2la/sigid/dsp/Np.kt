package br.pp2la.sigid.dsp

import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/** Funções no estilo numpy/scipy, com a mesma semântica usada em sigfeatures.py. */
object Np {
    const val EPS = 1e-12

    fun mean(a: DoubleArray): Double { var s = 0.0; for (v in a) s += v; return s / a.size }

    fun std(a: DoubleArray): Double {
        val m = mean(a); var s = 0.0
        for (v in a) { val d = v - m; s += d * d }
        return sqrt(s / a.size)
    }

    fun argmax(a: DoubleArray, from: Int = 0, to: Int = a.size): Int {
        var b = from; var bv = a[from]
        for (i in from + 1 until to) if (a[i] > bv) { bv = a[i]; b = i }
        return b
    }

    /** Mediana como numpy (média dos dois centrais se n par), via quickselect O(n). */
    fun median(a: DoubleArray): Double {
        val n = a.size
        if (n < 64) {
            val s = a.copyOf(); s.sort()
            return if (n % 2 == 1) s[n / 2] else 0.5 * (s[n / 2 - 1] + s[n / 2])
        }
        val s = a.copyOf()
        val hi = select(s, n / 2)
        if (n % 2 == 1) return hi
        // o elemento n/2-1 é o máximo da metade esquerda após o select
        var lo = s[0]
        for (i in 1 until n / 2) if (s[i] > lo) lo = s[i]
        return 0.5 * (lo + hi)
    }

    /** k-ésimo menor (in-place, particiona s). */
    private fun select(s: DoubleArray, k: Int): Double {
        var l = 0; var r = s.size - 1
        val rnd = java.util.Random(12345)
        while (true) {
            if (l == r) return s[l]
            val pIdx = l + rnd.nextInt(r - l + 1)
            val pivot = s[pIdx]
            // partição de 3 vias
            var lt = l; var gt = r; var i = l
            while (i <= gt) {
                val v = s[i]
                when {
                    v < pivot -> { s[i] = s[lt]; s[lt] = v; lt++; i++ }
                    v > pivot -> { s[i] = s[gt]; s[gt] = v; gt-- }
                    else -> i++
                }
            }
            when {
                k < lt -> r = lt - 1
                k > gt -> l = gt + 1
                else -> return pivot
            }
        }
    }

    /** numpy.percentile (interpolação linear) sobre um array já ordenado. */
    fun percentileSorted(s: DoubleArray, q: Double): Double {
        val h = (s.size - 1) * q / 100.0
        val lo = floor(h).toInt()
        val hi = min(lo + 1, s.size - 1)
        val t = h - lo
        val a = s[lo]; val b = s[hi]
        return if (t >= 0.5) b - (b - a) * (1 - t) else a + (b - a) * t
    }

    /** numpy.linspace */
    fun linspace(start: Double, stop: Double, num: Int): DoubleArray {
        val out = DoubleArray(num)
        val step = (stop - start) / (num - 1)
        for (i in 0 until num) out[i] = start + i * step
        out[num - 1] = stop
        return out
    }

    /** numpy.geomspace */
    fun geomspace(start: Double, stop: Double, num: Int): DoubleArray {
        val ls = linspace(log10(start), log10(stop), num)
        val out = DoubleArray(num) { 10.0.pow(ls[it]) }
        out[0] = start; out[num - 1] = stop
        return out
    }

    /** numpy.histogram com bordas explícitas: [e_i, e_i+1), último bin fechado. */
    fun histogram(x: DoubleArray, edges: DoubleArray): DoubleArray {
        val nb = edges.size - 1
        val h = DoubleArray(nb)
        val first = edges[0]; val last = edges[nb]
        for (v in x) {
            if (v.isNaN() || v < first || v > last) continue
            if (v == last) { h[nb - 1] += 1.0; continue }
            var lo = 0; var hi = nb
            while (hi - lo > 1) { val mid = (lo + hi) ushr 1; if (edges[mid] <= v) lo = mid else hi = mid }
            h[lo] += 1.0
        }
        return h
    }

    /** numpy.searchsorted(a, v, side='left') */
    fun searchLeft(a: DoubleArray, v: Double): Int {
        var lo = 0; var hi = a.size
        while (lo < hi) { val mid = (lo + hi) ushr 1; if (a[mid] < v) lo = mid + 1 else hi = mid }
        return lo
    }

    /** numpy.convolve(a, ones(k)/k, mode='same') via soma acumulada. */
    fun movingAvgSame(a: DoubleArray, k: Int): DoubleArray {
        val n = a.size
        val off = (k - 1) / 2
        val pre = DoubleArray(n + 1)
        for (i in 0 until n) pre[i + 1] = pre[i] + a[i]
        val out = DoubleArray(n)
        for (i in 0 until n) {
            val hi = min(n - 1, i + off)            // índices i+off-j, j=0..k-1
            val lo = max(0, i + off - k + 1)
            out[i] = if (hi >= lo) (pre[hi + 1] - pre[lo]) / k else 0.0
        }
        return out
    }

    /** numpy.convolve(a, ones(k)/k, 'same') somando termo a termo (mesma ordem de soma do numpy p/ k pequeno). */
    fun movingAvgSameExact(a: DoubleArray, k: Int): DoubleArray {
        val n = a.size; val off = (k - 1) / 2; val w = 1.0 / k
        return DoubleArray(n) { i ->
            var s = 0.0
            for (j in 0 until k) { val idx = i + off - j; if (idx in 0 until n) s += a[idx] * w }
            s
        }
    }

    /** scipy.signal.medfilt (preenchimento com zeros). */
    fun medfilt(a: DoubleArray, k: Int): DoubleArray {
        val n = a.size; val h = k / 2
        val out = DoubleArray(n); val buf = DoubleArray(k)
        for (i in 0 until n) {
            for (j in 0 until k) { val idx = i - h + j; buf[j] = if (idx in 0 until n) a[idx] else 0.0 }
            buf.sort()
            out[i] = buf[h]
        }
        return out
    }

    /** scipy.signal.find_peaks(x, prominence=pmin) -> (índices, proeminências). */
    fun findPeaksProm(x: DoubleArray, pmin: Double): Pair<IntArray, DoubleArray> {
        val peaks = ArrayList<Int>()
        val n = x.size
        var i = 1
        val iMax = n - 1
        while (i < iMax) {
            if (x[i - 1] < x[i]) {
                var ahead = i + 1
                while (ahead < iMax && x[ahead] == x[i]) ahead++
                if (x[ahead] < x[i]) {
                    peaks.add((i + ahead - 1) / 2)
                    i = ahead
                }
            }
            i++
        }
        val pk = ArrayList<Int>(); val pr = ArrayList<Double>()
        for (p in peaks) {
            var j = p; var leftMin = x[p]
            while (j >= 0 && x[j] <= x[p]) { if (x[j] < leftMin) leftMin = x[j]; j-- }
            j = p; var rightMin = x[p]
            while (j <= n - 1 && x[j] <= x[p]) { if (x[j] < rightMin) rightMin = x[j]; j++ }
            val prom = x[p] - max(leftMin, rightMin)
            if (prom >= pmin) { pk.add(p); pr.add(prom) }
        }
        return pk.toIntArray() to pr.toDoubleArray()
    }

    fun kurt(v: DoubleArray): Double {
        val m = mean(v); var s2 = 0.0; var s4 = 0.0
        for (x in v) { val d = x - m; val d2 = d * d; s2 += d2; s4 += d2 * d2 }
        s2 = s2 / v.size + EPS
        return (s4 / v.size) / (s2 * s2)
    }

    /** np.correlate(s, s, 'full')[n-1 : n-1+maxLag+1] */
    fun autocorr(s: DoubleArray, maxLag: Int): DoubleArray {
        val n = s.size
        val out = DoubleArray(maxLag + 1)
        for (k in 0..min(maxLag, n - 1)) { var acc = 0.0; for (i in 0 until n - k) acc += s[i + k] * s[i]; out[k] = acc }
        return out
    }

    /** numpy.hanning (simétrica) */
    fun hanningSym(m: Int): DoubleArray =
        if (m == 1) doubleArrayOf(1.0) else DoubleArray(m) { 0.5 - 0.5 * kotlin.math.cos(2 * Math.PI * it / (m - 1)) }
}
