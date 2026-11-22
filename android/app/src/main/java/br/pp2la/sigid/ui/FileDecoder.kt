package br.pp2la.sigid.ui

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sin

/** Decodifica qualquer áudio suportado pelo Android (WAV, MP3, OGG, M4A, FLAC) para mono 16 kHz. */
object FileDecoder {
    fun decodeTo16k(ctx: Context, uri: Uri, maxSeconds: Int = 180): FloatArray {
        val ex = MediaExtractor()
        ex.setDataSource(ctx, uri, null)
        val track = (0 until ex.trackCount).firstOrNull {
            ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: error("o arquivo não tem faixa de áudio")
        ex.selectTrack(track)
        val fmt = ex.getTrackFormat(track)
        val mime = fmt.getString(MediaFormat.KEY_MIME)!!
        var sr = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var ch = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var pcmFloat = false
        var out = FloatArray(sr * 30)
        var count = 0
        fun push(v: Float) {
            if (count == out.size) out = out.copyOf(out.size * 2)
            out[count++] = v
        }
        fun addPcm(bytes: java.nio.ByteBuffer) {
            bytes.order(ByteOrder.LITTLE_ENDIAN)
            if (pcmFloat) {
                val fb = bytes.asFloatBuffer()
                while (fb.remaining() >= ch) { var s = 0f; repeat(ch) { s += fb.get() }; push(s / ch) }
            } else {
                val sb = bytes.asShortBuffer()
                while (sb.remaining() >= ch) { var s = 0f; repeat(ch) { s += sb.get() / 32768f }; push(s / ch) }
            }
        }
        val limit = { count >= sr.toLong() * maxSeconds }
        if (mime == "audio/raw") {
            if (fmt.containsKey(MediaFormat.KEY_PCM_ENCODING))
                pcmFloat = fmt.getInteger(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
            val buf = java.nio.ByteBuffer.allocate(1 shl 16)
            while (!limit()) {
                buf.clear()
                val n = ex.readSampleData(buf, 0)
                if (n < 0) break
                buf.limit(n); addPcm(buf)
                ex.advance()
            }
        } else {
            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(fmt, null, null, 0)
            codec.start()
            val info = MediaCodec.BufferInfo()
            var inEnd = false; var outEnd = false
            while (!outEnd && !limit()) {
                if (!inEnd) {
                    val ii = codec.dequeueInputBuffer(10000)
                    if (ii >= 0) {
                        val ib = codec.getInputBuffer(ii)!!
                        val n = ex.readSampleData(ib, 0)
                        if (n < 0) { codec.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inEnd = true }
                        else { codec.queueInputBuffer(ii, 0, n, ex.sampleTime, 0); ex.advance() }
                    }
                }
                val oi = codec.dequeueOutputBuffer(info, 10000)
                when {
                    oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val of = codec.outputFormat
                        sr = of.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        ch = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        if (of.containsKey(MediaFormat.KEY_PCM_ENCODING))
                            pcmFloat = of.getInteger(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
                    }
                    oi >= 0 -> {
                        val ob = codec.getOutputBuffer(oi)!!
                        ob.position(info.offset); ob.limit(info.offset + info.size)
                        addPcm(ob.slice())
                        codec.releaseOutputBuffer(oi, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outEnd = true
                    }
                }
            }
            codec.stop(); codec.release()
        }
        ex.release()
        val x = out.copyOf(count)
        return if (sr == 16000) x else resample(x, sr, 16000)
    }

    /** Reamostragem por sinc janelada (passa-baixas anti-aliasing incluso). */
    fun resample(x: FloatArray, srIn: Int, srOut: Int): FloatArray {
        val ratio = srOut.toDouble() / srIn
        val fc = min(1.0, ratio) * 0.95
        val half = (24 / fc).toInt()
        val nOut = (x.size * ratio).toInt()
        val y = FloatArray(nOut)
        for (n in 0 until nOut) {
            val t = n / ratio
            val c = floor(t).toInt()
            var acc = 0.0; var wsum = 0.0
            for (k in c - half + 1..c + half) {
                if (k < 0 || k >= x.size) continue
                val u = t - k
                val arg = PI * fc * u
                val sinc = if (kotlin.math.abs(arg) < 1e-9) 1.0 else sin(arg) / arg
                val w = 0.5 + 0.5 * cos(PI * u / half)
                val h = fc * sinc * w
                acc += x[k] * h; wsum += h
            }
            y[n] = if (wsum != 0.0) (acc / wsum).toFloat() else 0f
        }
        return y
    }
}
