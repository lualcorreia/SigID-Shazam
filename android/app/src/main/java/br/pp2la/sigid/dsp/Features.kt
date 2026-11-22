package br.pp2la.sigid.dsp

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Porte 1:1 de sigfeatures.extract() (Python). Mesmas 468 features, mesma ordem.
 * fb: banco mel 64x513 exportado do Python; lags: atrasos da autocorrelação de energia (T=247).
 */
class Features(private val fb: Array<DoubleArray>, private val lagsExported: IntArray) {
    companion object {
        const val FS = 16000
        const val NFFT = 1024
        const val HOP = 256
        const val NBIN = NFFT / 2 + 1
        const val N_FEATURES = 468
        const val WIN = 64000
        private const val EPS = 1e-12
        val DF = 1.0 / (NFFT * (1.0 / FS))
    }

    private val win = DoubleArray(NFFT) { 0.5 - 0.5 * cos(2.0 * PI * it / NFFT) } // scipy 'hann' periódica
    private val winSum = win.sum()
    private val f = DoubleArray(NBIN) { it * DF }
    private val fbLo = IntArray(fb.size) { m -> fb[m].indexOfFirst { it != 0.0 }.coerceAtLeast(0) }
    private val fbHi = IntArray(fb.size) { m -> fb[m].indexOfLast { it != 0.0 }.coerceAtLeast(0) }
    private val dctM = Array(20) { k ->
        DoubleArray(64) { n ->
            val s = if (k == 0) sqrt(1.0 / 64) else sqrt(2.0 / 64)
            s * cos(PI * k * (2 * n + 1) / (2.0 * 64))
        }
    }
    private val histRel40 = Np.linspace(-1500.0, 1500.0, 41)
    private val histRel30 = Np.linspace(-1500.0, 1500.0, 31)
    private val histAbs = DoubleArray(251) { it * 32.0 }

    private var marks: MutableList<Pair<String, Long>>? = null
    private fun mark(s: String) { marks?.add(s to System.nanoTime()) }

    /** Tempo por seção do extract (para otimização). */
    fun profile(x: FloatArray): String {
        marks = mutableListOf("início" to System.nanoTime())
        extract(x)
        val m = marks!!; marks = null
        return (1 until m.size).joinToString(", ") { "${m[it].first}=${(m[it].second - m[it - 1].second) / 1000000}ms" }
    }

    fun extract(input: FloatArray): FloatArray {
        var n = input.size
        val x = DoubleArray(max(n, FS))
        for (i in 0 until n) x[i] = input[i].toDouble()
        n = x.size
        val mu = Np.mean(x)
        for (i in 0 until n) x[i] -= mu
        var ss = 0.0
        for (v in x) ss += v * v
        val rms = sqrt(ss / n) + EPS
        for (i in 0 until n) x[i] /= rms

        val o = DoubleArray(N_FEATURES)
        var p = 0
        fun put(v: Double) { o[p++] = v }
        fun putAll(a: DoubleArray) { for (v in a) o[p++] = v }

        // ---------- STFT
        val T = (n - NFFT) / HOP + 1
        val P = Array(NBIN) { DoubleArray(T) }
        val re = DoubleArray(NFFT); val im = DoubleArray(NFFT)
        for (t in 0 until T) {
            val s0 = t * HOP
            for (i in 0 until NFFT) { re[i] = x[s0 + i] * win[i]; im[i] = 0.0 }
            Fft.fft2(re, im)
            for (k in 0 until NBIN) {
                val zr = re[k] / winSum; val zi = im[k] / winSum
                P[k][t] = zr * zr + zi * zi + EPS
            }
        }

        mark("stft")
        // mel / MFCC
        val nm = fb.size
        val logM = Array(nm) { DoubleArray(T) }
        for (m in 0 until nm) {
            val row = fb[m]
            for (t in 0 until T) {
                var acc = 0.0
                for (k in fbLo[m]..fbHi[m]) acc += row[k] * P[k][t]
                logM[m][t] = 10.0 * log10(acc + EPS)
            }
        }
        val lmMean = DoubleArray(nm) { Np.mean(logM[it]) }
        val lmMax = lmMean.max()
        putAll(DoubleArray(nm) { lmMean[it] - lmMax })
        putAll(DoubleArray(nm) { Np.std(logM[it]) })
        val mfcc = Array(20) { k ->
            DoubleArray(T) { t -> var acc = 0.0; for (m in 0 until nm) acc += dctM[k][m] * logM[m][t]; acc }
        }
        putAll(DoubleArray(20) { Np.mean(mfcc[it]) })
        putAll(DoubleArray(20) { Np.std(mfcc[it]) })
        putAll(DoubleArray(20) { k -> if (T > 1) Np.std(DoubleArray(T - 1) { mfcc[k][it + 1] - mfcc[k][it] }) else 0.0 })

        mark("mel")
        // espectro médio
        val S = DoubleArray(NBIN) { Np.mean(P[it]) }
        val Sdb = DoubleArray(NBIN) { 10.0 * log10(S[it]) }
        val sdbMax = Sdb.max()
        for (k in 0 until NBIN) Sdb[k] -= sdbMax
        for (th in doubleArrayOf(-6.0, -10.0, -20.0, -30.0)) {
            var cnt = 0; var first = -1; var last = -1
            for (k in 0 until NBIN) if (Sdb[k] > th) { cnt++; if (first < 0) first = k; last = k }
            put(cnt.toDouble() / NBIN)
            put(if (cnt > 0) (last - first) * DF / 1000 else 0.0)
        }
        val sSum = S.sum()
        val pr = DoubleArray(NBIN) { S[it] / sSum }
        var cen = 0.0
        for (k in 0 until NBIN) cen += f[k] * pr[k]
        var sp2 = 0.0
        for (k in 0 until NBIN) sp2 += (f[k] - cen) * (f[k] - cen) * pr[k]
        val spread = sqrt(sp2)
        val cums = DoubleArray(NBIN)
        var c = 0.0
        for (k in 0 until NBIN) { c += pr[k]; cums[k] = c }
        val rolls = doubleArrayOf(0.05, 0.15, 0.5, 0.85, 0.95).map { q -> f[min(Np.searchLeft(cums, q), NBIN - 1)] }
        var lsum = 0.0
        for (v in S) lsum += ln(v)
        val flat = exp(lsum / NBIN) / Np.mean(S)
        put(cen / 1000); put(spread / 1000); put(skewW(pr, cen)); put(log10(kurtW(pr, cen) + EPS))
        for (r in rolls) put(r / 1000)
        put(flat); put(log10(flat + EPS))

        // banda principal
        val sm = Np.movingAvgSameExact(Sdb, 5)
        val k0 = Np.argmax(sm)
        var lo = k0
        while (lo > 0 && sm[lo - 1] > -25) lo--
        var hi = k0
        while (hi < NBIN - 1 && sm[hi + 1] > -25) hi++
        var fLo = f[lo]; var fHi = f[hi]
        if (fHi - fLo < 300) {
            val cc = 0.5 * (fLo + fHi)
            fLo = max(cc - 150, 0.0); fHi = min(cc + 150, FS / 2.0)
        }
        put(fLo / 1000); put(fHi / 1000); put((fHi - fLo) / 1000); put(f[k0] / 1000)

        mark("forma")
        // estrutura de tons
        val med31 = Np.medfilt(Sdb, 31)
        val det = DoubleArray(NBIN) { Sdb[it] - med31[it] }
        val (pk, prom) = Np.findPeaksProm(Sdb, 3.0)
        val n6 = prom.count { it > 6 }; val n12 = prom.count { it > 12 }; val n20 = prom.count { it > 20 }
        val order = prom.indices.sortedWith(compareBy { prom[it] }).reversed().take(4)
        val topf = order.map { f[pk[it]] }.sorted()
        val tf = DoubleArray(4); for (i in topf.indices) tf[i] = topf[i] / 1000
        val spc = DoubleArray(3); for (i in 0 until topf.size - 1) spc[i] = (topf[i + 1] - topf[i]) / 1000
        put(n6.toDouble()); put(n12.toDouble()); put(n20.toDouble()); put(ln1p(n6.toDouble()))
        put(if (prom.isNotEmpty()) prom.max() else 0.0)
        putAll(tf); putAll(spc)

        val dm = Np.mean(det)
        val sdet = DoubleArray(NBIN) { det[it] - dm }
        val ac = Np.autocorr(sdet, 128)
        val ac0 = ac[0] + EPS
        for (g in 0 until 32) {
            var acc = 0.0
            for (j in 1..4) acc += ac[4 * g + j] / ac0
            put(acc / 4)
        }

        mark("tons")
        // por quadro
        val frFlat = DoubleArray(T); val frCen = DoubleArray(T); val colSum = DoubleArray(T)
        for (t in 0 until T) {
            var sl = 0.0; var sa = 0.0; var sf = 0.0
            for (k in 0 until NBIN) { val v = P[k][t]; sl += ln(v); sa += v; sf += f[k] * v }
            frFlat[t] = exp(sl / NBIN) / (sa / NBIN)
            frCen[t] = sf / sa
            colSum[t] = sa
        }
        val flux = if (T > 1) DoubleArray(T - 1) { t ->
            var acc = 0.0
            for (k in 0 until NBIN) { val d = P[k][t + 1] / colSum[t + 1] - P[k][t] / colSum[t]; acc += d * d }
            sqrt(acc)
        } else DoubleArray(1)
        val kb0 = f.indexOfFirst { it > 80 }
        val kb1 = f.indexOfLast { it < FS / 2.0 - 80 }
        val pkf = DoubleArray(T) { t ->
            var b = kb0; var bv = P[kb0][t]
            for (k in kb0 + 1..kb1) if (P[k][t] > bv) { bv = P[k][t]; b = k }
            f[b]
        }
        val medPk = Np.median(pkf)
        val rel = DoubleArray(T) { pkf[it] - medPk }
        val h = Np.histogram(rel, histRel40).also { a -> for (i in a.indices) a[i] = a[i] / T.toDouble() }
        val habs = Np.histogram(pkf, histAbs).also { a -> for (i in a.indices) a[i] = a[i] / T.toDouble() }
        var ent = 0.0
        for (v in habs) if (v > 0) ent -= v * ln(v)
        val dpk = DoubleArray(max(T - 1, 0)) { pkf[it + 1] - pkf[it] }
        val nz = dpk.filter { abs(it) > 0 }
        put(Np.mean(frFlat)); put(Np.std(frFlat)); put(Np.std(frCen) / 1000)
        put(Np.mean(flux)); put(Np.std(flux)); put(ent)
        put(habs.count { it > 0.05 }.toDouble()); put(habs.count { it > 0.15 }.toDouble())
        put(if (dpk.isNotEmpty()) dpk.count { abs(it) <= 2 * DF }.toDouble() / dpk.size else 1.0)
        put(if (nz.isNotEmpty()) abs(nz.sumOf { Math.signum(it) } / nz.size) else 0.0)
        put(if (dpk.isNotEmpty()) Np.median(DoubleArray(dpk.size) { abs(dpk[it]) }) / 1000 else 0.0)
        put(1 - h.sum())
        putAll(h)

        mark("quadros")
        // estrutura temporal lenta
        val e = DoubleArray(T) { 10.0 * log10(colSum[it]) }
        val es = e.copyOf().also { it.sort() }
        val loE = Np.percentileSorted(es, 10.0); val hiE = Np.percentileSorted(es, 90.0)
        val thr = (loE + hiE) / 2
        var trans = 0
        for (t in 1 until T) if ((e[t] > thr) != (e[t - 1] > thr)) trans++
        val onMean = e.count { it > thr }.toDouble() / T
        val dur = n.toDouble() / FS
        val eMax = e.max()
        val el = DoubleArray(T) { 10.0.pow((e[it] - eMax) / 10) }
        val elm = Np.mean(el)
        for (t in 0 until T) el[t] -= elm
        val acf = Np.autocorr(el, T - 1)
        val acf0 = acf[0] + EPS
        val lags = if (T == 247) lagsExported else
            Np.geomspace(1.0, max(2, T / 2).toDouble(), 16).map { it.toInt() }.distinct().sorted().toIntArray()
        val a16 = DoubleArray(16)
        for (i in lags.indices) a16[i] = acf[min(lags[i], acf.size - 1)] / acf0
        put(hiE - loE); put(Np.percentileSorted(es, 99.0) - Np.percentileSorted(es, 1.0))
        put(onMean); put(ln1p(trans / dur)); put(Np.kurt(el)); putAll(a16)
        val frRate = FS.toDouble() / HOP
        val em = Np.mean(e)
        putAll(lineSpectrum(DoubleArray(T) { e[it] - em }, frRate, 0.3, frRate / 2 - 0.5, 16).first)

        mark("energia")
        // sinal analítico limitado à banda principal
        val (Xr, Xi) = Fft.fft(x, DoubleArray(n))
        val dval = 1.0 / (n * (1.0 / FS))
        val ar0 = DoubleArray(n); val ai0 = DoubleArray(n)
        val npos = (n - 1) / 2 + 1
        for (k in 0 until n) {
            val fk = (if (k < npos) k else k - n) * dval
            if (fk >= fLo && fk <= fHi) { ar0[k] = 2 * Xr[k]; ai0[k] = 2 * Xi[k] }
        }
        val (ar, ai) = Fft.ifft(ar0, ai0)
        val env = DoubleArray(n) { sqrt(ar[it] * ar[it] + ai[it] * ai[it]) + EPS }
        val env2 = DoubleArray(n) { env[it] * env[it] }
        put(Np.std(env) / Np.mean(env)); put(log10(Np.kurt(env)))
        put(min(10 * log10(env2.max() / Np.mean(env2)), 30.0))
        val (l1, t1) = lineSpectrum(env2, FS.toDouble(), 2.0, 4000.0, 40)
        putAll(l1); putAll(t1)

        mark("analitico+env")
        // frequência instantânea
        val ph = DoubleArray(n - 1) { i ->
            val r = ar[i + 1] * ar[i] + ai[i + 1] * ai[i]
            val q = ai[i + 1] * ar[i] - ar[i + 1] * ai[i]
            atan2(q, r) * FS / (2 * PI)
        }
        val ifr = Np.movingAvgSame(ph, 8)
        val medIf = Np.median(ifr)
        val relIf = DoubleArray(ifr.size) { ifr[it] - medIf }
        val hif = Np.histogram(relIf, histRel30).also { a -> for (i in a.indices) a[i] = a[i] / relIf.size.toDouble() }
        putAll(hif)
        val sdIf = Np.std(relIf)
        put(sdIf / 1000); put(if (sdIf > 0) Np.kurt(relIf) else 0.0)
        val (l2, t2) = lineSpectrum(ifr, FS.toDouble(), 2.0, 4000.0, 40)
        putAll(l2); putAll(t2)

        mark("freqinst")
        // potências do sinal analítico
        var pa = 0.0
        for (i in 0 until n) pa += ar[i] * ar[i] + ai[i] * ai[i]
        val nrm = sqrt(pa / n) + EPS
        var pr2 = DoubleArray(n) { ar[it] / nrm }
        var pi2 = DoubleArray(n) { ai[it] / nrm }
        for (pw in intArrayOf(1, 2, 4, 8)) {
            if (pw > 1) { // eleva ao quadrado: a^1 -> a^2 -> a^4 -> a^8
                val nr = DoubleArray(n); val ni = DoubleArray(n)
                for (i in 0 until n) { nr[i] = pr2[i] * pr2[i] - pi2[i] * pi2[i]; ni[i] = 2 * pr2[i] * pi2[i] }
                pr2 = nr; pi2 = ni
            }
            val (sr, si) = Fft.fft(pr2, pi2)
            val mag = DoubleArray(n) { sqrt(sr[it] * sr[it] + si[it] * si[it]) }
            put(min(20 * log10(mag.max() / (Np.median(mag) + EPS)), 80.0))
        }

        mark("potencias")
        // zcr
        var zc = 0
        for (i in 1 until n) if (signbit(x[i]) != signbit(x[i - 1])) zc++
        put(zc.toDouble() / (n - 1))

        mark("zcr")
        check(p == N_FEATURES) { "features: $p != $N_FEATURES" }
        return FloatArray(N_FEATURES) { i ->
            val v = o[i]
            when {
                v.isNaN() -> 0f
                v == Double.POSITIVE_INFINITY -> 80f
                v == Double.NEGATIVE_INFINITY -> -80f
                else -> v.toFloat()
            }
        }
    }

    private fun signbit(v: Double) = v < 0 || (v == 0.0 && 1.0 / v < 0)

    private fun skewW(p: DoubleArray, m: Double): Double {
        var s2 = 0.0
        for (k in p.indices) s2 += (f[k] - m) * (f[k] - m) * p[k]
        val s = sqrt(s2) + EPS
        var acc = 0.0
        for (k in p.indices) { val z = (f[k] - m) / s; acc += z * z * z * p[k] }
        return acc
    }

    private fun kurtW(p: DoubleArray, m: Double): Double {
        var s2 = 0.0
        for (k in p.indices) s2 += (f[k] - m) * (f[k] - m) * p[k]
        val s = sqrt(s2) + EPS
        var acc = 0.0
        for (k in p.indices) { val z = (f[k] - m) / s; acc += z * z * z * z * p[k] }
        return acc
    }

    /** _line_spectrum_feats: espectro de um sinal auxiliar em bins log, dB relativo à mediana. */
    private fun lineSpectrum(sigIn: DoubleArray, fs: Double, fmin: Double, fmax: Double, nb: Int):
            Pair<DoubleArray, DoubleArray> {
        val m = sigIn.size
        val mu = Np.mean(sigIn)
        val w = Np.hanningSym(m)
        val S = Fft.rfftMag(DoubleArray(m) { (sigIn[it] - mu) * w[it] })
        val v = 1.0 / (m * (1.0 / fs))
        val fr = DoubleArray(S.size) { it * v }
        val bandIdx = fr.indices.filter { fr[it] >= fmin && fr[it] <= fmax }
        val sb = DoubleArray(bandIdx.size) { S[bandIdx[it]] }
        val med = Np.median(sb) + EPS
        val edges = Np.geomspace(fmin, fmax, nb + 1)
        val idx = IntArray(nb + 1) { Np.searchLeft(fr, edges[it]) }
        val pooled = DoubleArray(nb) { i ->
            val a = idx[i]; val b = max(idx[i + 1], a + 1)
            var mx = 0.0
            if (a < S.size) { mx = S[a]; for (k in a + 1 until min(b, S.size)) if (S[k] > mx) mx = S[k] }
            (20 * log10(mx / med + EPS)).coerceIn(-20.0, 60.0)
        }
        val kb = Np.argmax(sb)
        val top = doubleArrayOf(log10(fr[bandIdx[kb]]), min(20 * log10(sb[kb] / med), 80.0))
        return pooled to top
    }
}
