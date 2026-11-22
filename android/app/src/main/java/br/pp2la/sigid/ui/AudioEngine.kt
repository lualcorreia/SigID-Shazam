package br.pp2la.sigid.ui

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.media.audiofx.AcousticEchoCanceler

/**
 * Captura do microfone a 16 kHz mono. Usa a fonte UNPROCESSED quando o aparelho suporta
 * (sem AGC/supressão de ruído, que alterariam a assinatura do sinal) e desliga os efeitos.
 */
class AudioEngine(private val ctx: Context) {
    companion object {
        const val FS = 16000
        const val RING_S = 30
    }

    @Volatile private var running = false
    private var thread: Thread? = null
    private val ring = FloatArray(FS * RING_S)
    private var writePos = 0L
    private val lock = Any()
    var sourceName = ""
        private set

    /** Chamado na thread de áudio com cada bloco novo (para o waterfall). */
    var onBlock: ((FloatArray) -> Unit)? = null

    fun supportsUnprocessed(): Boolean {
        val am = ctx.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return am.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
    }

    @SuppressLint("MissingPermission")
    fun start(preferred: Int): Boolean {
        stop()
        val minBuf = AudioRecord.getMinBufferSize(FS, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val sources = buildList {
            add(preferred)
            if (supportsUnprocessed()) add(MediaRecorder.AudioSource.UNPROCESSED)
            add(MediaRecorder.AudioSource.VOICE_RECOGNITION)
            add(MediaRecorder.AudioSource.MIC)
        }.distinct()
        var rec: AudioRecord? = null
        for (s in sources) {
            try {
                val r = AudioRecord(s, FS, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    maxOf(minBuf * 4, FS))
                if (r.state == AudioRecord.STATE_INITIALIZED) {
                    rec = r
                    sourceName = when (s) {
                        MediaRecorder.AudioSource.UNPROCESSED -> "microfone sem processamento"
                        MediaRecorder.AudioSource.VOICE_RECOGNITION -> "microfone (reconhecimento de voz)"
                        else -> "microfone padrão"
                    }
                    break
                } else r.release()
            } catch (_: Exception) { }
        }
        val r = rec ?: return false
        try { if (AutomaticGainControl.isAvailable()) AutomaticGainControl.create(r.audioSessionId)?.setEnabled(false) } catch (_: Exception) {}
        try { if (NoiseSuppressor.isAvailable()) NoiseSuppressor.create(r.audioSessionId)?.setEnabled(false) } catch (_: Exception) {}
        try { if (AcousticEchoCanceler.isAvailable()) AcousticEchoCanceler.create(r.audioSessionId)?.setEnabled(false) } catch (_: Exception) {}
        running = true
        synchronized(lock) { writePos = 0; ring.fill(0f) }
        thread = Thread({
            val buf = ShortArray(1024)
            r.startRecording()
            while (running) {
                val n = r.read(buf, 0, buf.size)
                if (n <= 0) continue
                val block = FloatArray(n) { buf[it] / 32768f }
                synchronized(lock) {
                    for (v in block) { ring[(writePos % ring.size).toInt()] = v; writePos++ }
                }
                onBlock?.invoke(block)
            }
            try { r.stop() } catch (_: Exception) {}
            r.release()
        }, "sigid-audio").also { it.priority = Thread.MAX_PRIORITY; it.start() }
        return true
    }

    fun stop() {
        running = false
        thread?.join(500)
        thread = null
    }

    val isRunning get() = running

    /** Segundos de áudio disponíveis no buffer. */
    fun available(): Double = synchronized(lock) { minOf(writePos, ring.size.toLong()).toDouble() / FS }

    /** Últimos n amostras (ordem cronológica). */
    fun last(n: Int): FloatArray = synchronized(lock) {
        val m = minOf(n.toLong(), writePos, ring.size.toLong()).toInt()
        FloatArray(m) { ring[((writePos - m + it) % ring.size).toInt()] }
    }
}
