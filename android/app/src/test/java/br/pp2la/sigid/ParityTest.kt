package br.pp2la.sigid

import br.pp2la.sigid.model.BinArrays
import br.pp2la.sigid.model.SigModel
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs
import kotlin.math.max

/** Compara o porte Kotlin com o Python (vetores gerados por export_android.py). */
class ParityTest {
    private fun asset(name: String) = File("src/main/assets/$name")

    @Test
    fun featuresAndProbabilitiesMatchPython() {
        val vec = BinArrays.read(javaClass.classLoader!!.getResourceAsStream("vectors.bin"))
        val x = vec.getValue("x"); val feat = vec.getValue("feat"); val pRef = vec.getValue("p"); val pmRef = vec.getValue("pm")
        val nVec = x.shape[0]; val n = x.shape[1]; val nf = feat.shape[1]; val nc = pRef.shape[1]; val nm = pmRef.shape[1]
        val arrays = BinArrays.read(asset("model.bin").inputStream())
        val nMlp = arrays.keys.count { it.matches(Regex("mlp\\d+_W0")) }
        val nLayers = arrays.keys.count { it.startsWith("mlp0_W") }
        val nMod = arrays.keys.count { it.startsWith("mod_W") }
        val model = SigModel(arrays, nMlp, nLayers, nMod, nc, 2.0, 0.65, 7)
        val xs = x.floats(); val fr = feat.floats(); val pr = pRef.floats(); val pmr = pmRef.floats()
        var worstFeat = 0.0; var nBad = 0; var top1Ok = 0; var top1ModOk = 0; var maxDp = 0.0
        for (v in 0 until nVec) {
            val xi = FloatArray(n) { xs[v * n + it] }
            val t0 = System.nanoTime()
            val f = model.features.extract(xi)
            val t1 = System.nanoTime()
            val (p, pm) = model.proba(f)
            val t2 = System.nanoTime()
            for (j in 0 until nf) {
                val ref = fr[v * nf + j].toDouble()
                val err = abs(f[j] - ref) / max(1.0, abs(ref))
                if (err > worstFeat) worstFeat = err
                if (err > 1e-3) { nBad++; if (nBad <= 25) println("  vetor $v feature $j: kotlin=${f[j]} python=$ref") }
            }
            val argK = p.indices.maxBy { p[it] }; val argP = (0 until nc).maxBy { pr[v * nc + it] }
            if (argK == argP) top1Ok++
            val amK = pm.indices.maxBy { pm[it] }; val amP = (0 until nm).maxBy { pmr[v * nm + it] }
            if (amK == amP) top1ModOk++
            for (j in 0 until nc) maxDp = max(maxDp, abs(p[j] - pr[v * nc + j]))
            println("vetor $v: features ${(t1 - t0) / 1e6} ms, inferência ${(t2 - t1) / 1e6} ms, " +
                "top1 kotlin=$argK python=$argP p=${"%.3f".format(p[argK])}/${"%.3f".format(pr[v * nc + argP])}")
        }
        println("pior erro relativo de feature = $worstFeat ; features > 1e-3: $nBad de ${nVec * nf}")
        println("top-1 sinal igual: $top1Ok/$nVec ; top-1 modulação igual: $top1ModOk/$nVec ; max |dp| = $maxDp")
        // tolerância: casos mal condicionados (tom puro sem ruído, fase em silêncio) divergem em poucas features
        assertTrue("features divergentes demais: $nBad", nBad <= nVec * nf * 0.03)
        assertTrue("top-1 divergente", top1Ok >= nVec - 1)
        assertTrue("modulação divergente", top1ModOk >= nVec - 1)
    }

    @Test
    fun benchmark() {
        val rnd = java.util.Random(1)
        val x = DoubleArray(64000) { rnd.nextGaussian() }
        repeat(3) { br.pp2la.sigid.dsp.Fft.fft(x, DoubleArray(64000)) }
        var t = System.nanoTime()
        repeat(10) { br.pp2la.sigid.dsp.Fft.fft(x, DoubleArray(64000)) }
        println("FFT 64000 (Bluestein): ${(System.nanoTime() - t) / 1e7} ms")
        val y = DoubleArray(131072) { rnd.nextGaussian() }
        t = System.nanoTime()
        repeat(10) { br.pp2la.sigid.dsp.Fft.fft2(y.copyOf(), DoubleArray(131072)) }
        println("FFT 131072 radix-2: ${(System.nanoTime() - t) / 1e7} ms")
        t = System.nanoTime()
        repeat(10) { br.pp2la.sigid.dsp.Np.median(x) }
        println("mediana 64000: ${(System.nanoTime() - t) / 1e7} ms")
        val arrays = BinArrays.read(asset("model.bin").inputStream())
        val model = SigModel(arrays, 3, 3, 2, arrays.getValue("mlp0_b2").size, 2.0, 0.65, 7)
        val xf = FloatArray(64000) { x[it].toFloat() }
        repeat(3) { model.features.extract(xf) }
        t = System.nanoTime()
        repeat(5) { model.features.extract(xf) }
        println("extract (aquecido): ${(System.nanoTime() - t) / 5e6} ms")
        val prof = model.features.profile(xf)
        println("perfil: $prof")
    }
}
