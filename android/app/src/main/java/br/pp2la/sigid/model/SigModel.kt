package br.pp2la.sigid.model

import br.pp2la.sigid.dsp.Features
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

/** Camada densa (W: in x out, como sklearn coefs_). */
class Dense(val W: FloatArray, val b: FloatArray, val nIn: Int, val nOut: Int)

class Mlp(val layers: List<Dense>) {
    /** sklearn MLPClassifier.predict_proba: ReLU nas ocultas, softmax na saída. */
    fun proba(z: FloatArray): DoubleArray {
        var a = DoubleArray(z.size) { z[it].toDouble() }
        for ((li, L) in layers.withIndex()) {
            val o = DoubleArray(L.nOut) { L.b[it].toDouble() }
            for (i in 0 until L.nIn) {
                val ai = a[i]
                if (ai == 0.0) continue
                val base = i * L.nOut
                for (j in 0 until L.nOut) o[j] += ai * L.W[base + j]
            }
            if (li < layers.size - 1) for (j in o.indices) if (o[j] < 0) o[j] = 0.0
            a = o
        }
        val mx = a.max()
        var s = 0.0
        for (j in a.indices) { a[j] = exp(a[j] - mx); s += a[j] }
        for (j in a.indices) a[j] /= s
        return a
    }
}

/** Ensemble exportado de sigid_core.fit_model: MLPs (com temperatura) + kNN cosseno em PCA + modulação. */
class SigModel(private val arrays: Map<String, NdArray>, nMlp: Int, nLayers: Int, nModLayers: Int,
               val nClasses: Int, private val T: Double, private val wMlp: Double, private val k: Int) {
    val features = Features(
        arrays.getValue("fb").let { a -> Array(a.shape[0]) { r -> DoubleArray(a.shape[1]) { a.d!![r * a.shape[1] + it] } } },
        arrays.getValue("lags").ints())
    private val mean = arrays.getValue("scaler_mean").doubles()
    private val scale = arrays.getValue("scaler_scale").doubles()
    private fun dense(w: String, b: String) = arrays.getValue(w).let { W ->
        Dense(W.floats(), arrays.getValue(b).floats(), W.shape[0], W.shape[1])
    }
    private val mlps = List(nMlp) { m -> Mlp(List(nLayers) { l -> dense("mlp${m}_W$l", "mlp${m}_b$l") }) }
    val mod = Mlp(List(nModLayers) { l -> dense("mod_W$l", "mod_b$l") })
    private val pcaMean = arrays.getValue("pca_mean").floats()
    private val pca = arrays.getValue("pca_comp").let { it.floats() to it.shape }
    private val nPc = pca.second[0]
    private val knnX = arrays.getValue("knn_X").floats()
    private val knnY = arrays.getValue("knn_y").ints()
    private val nTrain = knnY.size
    private val knnNorm = FloatArray(nTrain) { r ->
        var s = 0.0
        for (j in 0 until nPc) { val v = knnX[r * nPc + j].toDouble(); s += v * v }
        sqrt(s).toFloat()
    }

    fun standardize(F: FloatArray): FloatArray = FloatArray(F.size) {
        ((F[it] - mean[it]) / scale[it]).coerceIn(-8.0, 8.0).toFloat()
    }

    /** Retorna (p_sinais, p_modulação) para um vetor de features. */
    fun proba(F: FloatArray): Pair<DoubleArray, DoubleArray> {
        val z = standardize(F)
        val pm = DoubleArray(nClasses)
        for (m in mlps) {
            val q = m.proba(z)
            var s = 0.0
            for (j in q.indices) { q[j] = max(q[j], 1e-12).coerceAtMost(1.0).pow(1.0 / T); s += q[j] }
            for (j in q.indices) pm[j] += q[j] / s / mlps.size
        }
        val pk = knnProba(z)
        val p = DoubleArray(nClasses) { wMlp * pm[it] + (1 - wMlp) * pk[it] }
        val pmod = mod.proba(FloatArray(z.size) { z[it] })
        return p to pmod
    }

    private fun knnProba(z: FloatArray): DoubleArray {
        // PCA (whiten=False): (z - mean) @ components.T
        val q = DoubleArray(nPc)
        val comp = pca.first; val nf = pca.second[1]
        for (c in 0 until nPc) {
            var acc = 0.0
            for (j in 0 until nf) acc += (z[j] - pcaMean[j]).toDouble() * comp[c * nf + j]
            q[c] = acc
        }
        var qn = 0.0
        for (v in q) qn += v * v
        qn = sqrt(qn)
        // k menores distâncias cosseno
        val bestD = DoubleArray(k) { Double.MAX_VALUE }
        val bestI = IntArray(k) { -1 }
        for (r in 0 until nTrain) {
            var dot = 0.0
            val base = r * nPc
            for (j in 0 until nPc) dot += q[j] * knnX[base + j]
            val den = qn * knnNorm[r]
            val d = if (den > 0) (1.0 - dot / den).coerceIn(0.0, 2.0) else 1.0
            if (d < bestD[k - 1]) {
                var pos = k - 1
                while (pos > 0 && bestD[pos - 1] > d) { bestD[pos] = bestD[pos - 1]; bestI[pos] = bestI[pos - 1]; pos-- }
                bestD[pos] = d; bestI[pos] = r
            }
        }
        val out = DoubleArray(nClasses)
        val anyZero = bestD.any { it == 0.0 }
        for (i in 0 until k) {
            if (bestI[i] < 0) continue
            val w = if (anyZero) (if (bestD[i] == 0.0) 1.0 else 0.0) else 1.0 / bestD[i]
            out[knnY[bestI[i]]] += w
        }
        val s = out.sum()
        if (s > 0) for (j in out.indices) out[j] /= s
        return out
    }

    companion object { const val WIN = Features.WIN }
}
