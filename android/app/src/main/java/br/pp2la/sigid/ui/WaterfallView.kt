package br.pp2la.sigid.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import br.pp2la.sigid.dsp.Fft
import kotlin.math.log10
import kotlin.math.sqrt

/** Espectro (topo) + waterfall 0–8 kHz, alimentado pelos blocos do microfone. */
class WaterfallView(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    private val nfft = 1024
    private val nbin = nfft / 2
    private val rows = 260
    private val bmp = Bitmap.createBitmap(nbin, rows, Bitmap.Config.ARGB_8888)
    private val pixels = IntArray(nbin * rows)
    private val lut = IntArray(256) { inferno(it / 255f) }
    private val win = FloatArray(nfft) { (0.5 - 0.5 * Math.cos(2 * Math.PI * it / nfft)).toFloat() }
    private val acc = FloatArray(nfft)
    private var accN = 0
    private val spec = FloatArray(nbin)
    private var lo = -100f
    private var hi = -40f
    private val pLine = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#4FC3F7"); strokeWidth = 2f; style = Paint.Style.STROKE }
    private val pGrid = Paint().apply { color = Color.parseColor("#33FFFFFF"); strokeWidth = 1f }
    private val pText = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#AAFFFFFF"); textSize = 26f }
    private val dst = Rect()
    private val path = Path()
    private val lock = Any()

    /** Chamado da thread de áudio. */
    fun push(block: FloatArray) {
        var newRow = false
        synchronized(lock) {
            for (v in block) {
                acc[accN++] = v
                if (accN == nfft) {
                    computeRow()
                    // sobreposição de 50%
                    System.arraycopy(acc, nfft / 2, acc, 0, nfft / 2)
                    accN = nfft / 2
                    newRow = true
                }
            }
        }
        if (newRow) postInvalidate()
    }

    private fun computeRow() {
        val re = DoubleArray(nfft) { (acc[it] * win[it]).toDouble() }
        val im = DoubleArray(nfft)
        Fft.fft2(re, im)
        val db = FloatArray(nbin)
        for (k in 0 until nbin) {
            val m = sqrt(re[k] * re[k] + im[k] * im[k]) / (nfft / 4)
            db[k] = (20 * log10(m + 1e-9)).toFloat()
            spec[k] = 0.6f * spec[k] + 0.4f * db[k]
        }
        // níveis automáticos (lentos)
        val s = db.sortedArray()
        lo = 0.97f * lo + 0.03f * s[(nbin * 0.2).toInt()]
        hi = 0.97f * hi + 0.03f * maxOf(s[nbin - 3], lo + 25f)
        System.arraycopy(pixels, 0, pixels, nbin, nbin * (rows - 1))
        val span = (hi - lo).coerceAtLeast(10f)
        for (k in 0 until nbin) {
            val v = ((db[k] - lo) / span * 255).toInt().coerceIn(0, 255)
            pixels[k] = lut[v]
        }
    }

    override fun onDraw(c: Canvas) {
        val w = width.toFloat(); val h = height.toFloat()
        val specH = h * 0.28f
        synchronized(lock) {
            bmp.setPixels(pixels, 0, nbin, 0, 0, nbin, rows)
            dst.set(0, specH.toInt(), width, height)
            c.drawBitmap(bmp, null, dst, null)
            path.reset()
            val span = (hi - lo + 20).coerceAtLeast(10f)
            for (k in 0 until nbin) {
                val x = k * w / nbin
                val y = specH - ((spec[k] - (lo - 10)) / span).coerceIn(0f, 1f) * specH
                if (k == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
        }
        for (kHz in 1..7) {
            val x = kHz * 1000f / 8000f * w
            c.drawLine(x, 0f, x, specH, pGrid)
            c.drawText("$kHz", x + 4, 24f, pText)
        }
        c.drawPath(path, pLine)
        c.drawText("kHz", w - 52, 24f, pText)
    }

    private fun inferno(t: Float): Int {
        val stops = arrayOf(floatArrayOf(0f, 0f, 4f), floatArrayOf(40f, 11f, 84f), floatArrayOf(101f, 21f, 110f),
            floatArrayOf(159f, 42f, 99f), floatArrayOf(212f, 72f, 66f), floatArrayOf(245f, 125f, 21f),
            floatArrayOf(250f, 193f, 39f), floatArrayOf(252f, 255f, 164f))
        val x = t * (stops.size - 1)
        val i = x.toInt().coerceAtMost(stops.size - 2)
        val f = x - i
        val a = stops[i]; val b = stops[i + 1]
        return Color.rgb((a[0] + (b[0] - a[0]) * f).toInt(), (a[1] + (b[1] - a[1]) * f).toInt(), (a[2] + (b[2] - a[2]) * f).toInt())
    }
}
