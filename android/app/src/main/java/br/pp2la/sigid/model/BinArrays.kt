package br.pp2la.sigid.model

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Array n-dimensional lido do model.bin exportado pelo Python. */
class NdArray(val shape: IntArray, val f: FloatArray?, val d: DoubleArray?, val i: IntArray?) {
    val size get() = shape.fold(1) { a, b -> a * b }
    fun floats(): FloatArray = f ?: d?.let { a -> FloatArray(a.size) { a[it].toFloat() } }
        ?: i!!.let { a -> FloatArray(a.size) { a[it].toFloat() } }
    fun doubles(): DoubleArray = d ?: f?.let { a -> DoubleArray(a.size) { a[it].toDouble() } }
        ?: i!!.let { a -> DoubleArray(a.size) { a[it].toDouble() } }
    fun ints(): IntArray = i ?: floats().let { a -> IntArray(a.size) { a[it].toInt() } }
}

object BinArrays {
    fun read(input: InputStream): Map<String, NdArray> {
        val bytes = input.readBytes()
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val n = bb.int
        val out = LinkedHashMap<String, NdArray>()
        repeat(n) {
            val nl = bb.int
            val nb = ByteArray(nl); bb.get(nb)
            val name = String(nb, Charsets.UTF_8)
            val dt = bb.int
            val nd = bb.int
            val shape = IntArray(nd) { bb.int }
            val size = shape.fold(1) { a, b -> a * b }
            val arr = when (dt) {
                0 -> NdArray(shape, FloatArray(size) { bb.float }, null, null)
                1 -> NdArray(shape, FloatArray(size) { half(bb.short) }, null, null)
                2 -> NdArray(shape, null, null, IntArray(size) { bb.int })
                3 -> NdArray(shape, null, null, IntArray(size) { bb.short.toInt() })
                4 -> NdArray(shape, null, DoubleArray(size) { bb.double }, null)
                else -> error("dtype $dt")
            }
            out[name] = arr
        }
        return out
    }

    /** float16 IEEE -> float32 */
    fun half(h: Short): Float {
        val v = h.toInt() and 0xffff
        val sign = (v ushr 15) and 1
        val exp = (v ushr 10) and 0x1f
        val mant = v and 0x3ff
        val bits = when {
            exp == 0 -> {
                if (mant == 0) sign shl 31
                else { // subnormal
                    var m = mant; var e = -1
                    do { e++; m = m shl 1 } while (m and 0x400 == 0)
                    (sign shl 31) or ((127 - 15 - e) shl 23) or ((m and 0x3ff) shl 13)
                }
            }
            exp == 31 -> (sign shl 31) or (0xff shl 23) or (mant shl 13)
            else -> (sign shl 31) or ((exp - 15 + 127) shl 23) or (mant shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }
}
