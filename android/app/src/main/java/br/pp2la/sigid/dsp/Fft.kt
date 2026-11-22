package br.pp2la.sigid.dsp

import java.util.concurrent.ConcurrentHashMap
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** FFT complexa: radix-2 iterativa + Bluestein para qualquer tamanho (como numpy.fft). */
object Fft {
    private class Tw(val c: DoubleArray, val s: DoubleArray)

    private val twiddles = ConcurrentHashMap<Int, Tw>()
    private val chirps = ConcurrentHashMap<Long, Array<DoubleArray>>()

    private fun tw(n: Int): Tw = twiddles.getOrPut(n) {
        val c = DoubleArray(n / 2); val s = DoubleArray(n / 2)
        for (k in 0 until n / 2) { val a = -2.0 * PI * k / n; c[k] = cos(a); s[k] = sin(a) }
        Tw(c, s)
    }

    fun isPow2(n: Int) = n > 0 && (n and (n - 1)) == 0

    /** FFT radix-2 in-place (sem normalização). inverse = expoente positivo. */
    fun fft2(re: DoubleArray, im: DoubleArray, inverse: Boolean = false) {
        val n = re.size
        if (n <= 1) return
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        val t = tw(n)
        val sgn = if (inverse) -1.0 else 1.0
        // estágio len=2 (twiddle = 1)
        run {
            var i = 0
            while (i < n) {
                val xr = re[i + 1]; val xi = im[i + 1]
                re[i + 1] = re[i] - xr; im[i + 1] = im[i] - xi
                re[i] += xr; im[i] += xi
                i += 2
            }
        }
        // demais estágios com twiddles contíguos (amigável ao cache)
        val wr = DoubleArray(n / 2); val wi = DoubleArray(n / 2)
        var len = 4
        while (len <= n) {
            val half = len / 2
            val step = n / len
            for (k in 0 until half) { wr[k] = t.c[k * step]; wi[k] = sgn * t.s[k * step] }
            var i = 0
            while (i < n) {
                var a = i; var b = i + half
                for (k in 0 until half) {
                    val cr = wr[k]; val ci = wi[k]
                    val br = re[b]; val bi = im[b]
                    val xr = br * cr - bi * ci
                    val xi = br * ci + bi * cr
                    val ar = re[a]; val ai = im[a]
                    re[b] = ar - xr; im[b] = ai - xi
                    re[a] = ar + xr; im[a] = ai + xi
                    a++; b++
                }
                i += len
            }
            len = len shl 1
        }
    }

    /** FFT de qualquer tamanho; retorna novos arrays (re, im). Sem normalização. */
    fun fft(reIn: DoubleArray, imIn: DoubleArray, inverse: Boolean = false): Pair<DoubleArray, DoubleArray> {
        val n = reIn.size
        if (isPow2(n)) {
            val re = reIn.copyOf(); val im = imIn.copyOf()
            fft2(re, im, inverse)
            return re to im
        }
        // Bluestein: X_k = w_k * sum_j (x_j w_j) conj(w_{k-j}),  w_k = exp(-i pi k^2 / n)
        var m = 1
        while (m < 2 * n - 1) m = m shl 1
        val key = n.toLong() * 2 + (if (inverse) 1 else 0)
        val ch = chirps.getOrPut(key) {
            val wr = DoubleArray(n); val wi = DoubleArray(n)
            val sg = if (inverse) 1.0 else -1.0
            for (k in 0 until n) {
                val kk = (k.toLong() * k) % (2L * n)
                val a = sg * PI * kk / n
                wr[k] = cos(a); wi[k] = sin(a)
            }
            val br = DoubleArray(m); val bi = DoubleArray(m)
            br[0] = wr[0]; bi[0] = -wi[0]
            for (k in 1 until n) { br[k] = wr[k]; bi[k] = -wi[k]; br[m - k] = wr[k]; bi[m - k] = -wi[k] }
            fft2(br, bi, false)
            arrayOf(wr, wi, br, bi)
        }
        val wr = ch[0]; val wi = ch[1]; val br = ch[2]; val bi = ch[3]
        val ar = DoubleArray(m); val ai = DoubleArray(m)
        for (k in 0 until n) {
            ar[k] = reIn[k] * wr[k] - imIn[k] * wi[k]
            ai[k] = reIn[k] * wi[k] + imIn[k] * wr[k]
        }
        fft2(ar, ai, false)
        for (k in 0 until m) {
            val r = ar[k] * br[k] - ai[k] * bi[k]
            val i = ar[k] * bi[k] + ai[k] * br[k]
            ar[k] = r; ai[k] = i
        }
        fft2(ar, ai, true)
        val outR = DoubleArray(n); val outI = DoubleArray(n)
        for (k in 0 until n) {
            val cr = ar[k] / m; val ci = ai[k] / m
            outR[k] = cr * wr[k] - ci * wi[k]
            outI[k] = cr * wi[k] + ci * wr[k]
        }
        return outR to outI
    }

    /** numpy.fft.ifft (normalizado por 1/n). */
    fun ifft(re: DoubleArray, im: DoubleArray): Pair<DoubleArray, DoubleArray> {
        val (r, i) = fft(re, im, inverse = true)
        val n = re.size
        val inv = 1.0 / n
        for (k in 0 until n) { r[k] = r[k] * inv; i[k] = i[k] * inv }
        return r to i
    }

    /** |rfft(x)| (n/2+1 valores). */
    fun rfftMag(x: DoubleArray): DoubleArray {
        val n = x.size
        val (r, i) = fft(x, DoubleArray(n))
        val out = DoubleArray(n / 2 + 1)
        for (k in out.indices) out[k] = kotlin.math.sqrt(r[k] * r[k] + i[k] * i[k])
        return out
    }
}
