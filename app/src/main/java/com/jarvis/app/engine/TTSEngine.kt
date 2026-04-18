package com.jarvis.app.engine

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.Locale
import java.util.UUID
import kotlin.coroutines.resume

class TTSEngine(private val context: Context) {

    private var tts: TextToSpeech? = null
    private var isReady = false

    companion object {
        private const val TAG = "TTSEngine"
    }

    suspend fun init(): Boolean = suspendCancellableCoroutine { cont ->
        tts = TextToSpeech(context) { status ->
            isReady = (status == TextToSpeech.SUCCESS)
            if (isReady) {
                tts?.language = Locale.US
                tts?.setSpeechRate(1.0f)
                tts?.setPitch(0.95f)
                Log.i(TAG, "TTS ready")
            } else {
                Log.e(TAG, "TTS init failed: $status")
            }
            if (cont.isActive) cont.resume(isReady)
        }
    }

    fun speak(text: String, flush: Boolean = true) {
        if (!isReady || tts == null) { Log.w(TAG, "TTS not ready"); return }
        val queueMode = if (flush) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        tts?.speak(text, queueMode, null, UUID.randomUUID().toString())
    }

    suspend fun speakAndWait(text: String): Unit = suspendCancellableCoroutine { cont ->
        if (!isReady || tts == null) { cont.resume(Unit); return@suspendCancellableCoroutine }
        val uttId = UUID.randomUUID().toString()
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) {
                if (utteranceId == uttId && cont.isActive) cont.resume(Unit)
            }
            override fun onError(utteranceId: String?) {
                if (utteranceId == uttId && cont.isActive) cont.resume(Unit)
            }
        })
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, uttId)
    }

    fun stop() { tts?.stop() }

    fun setRate(rate: Float) { tts?.setSpeechRate(rate.coerceIn(0.5f, 2.0f)) }
    fun setPitch(pitch: Float) { tts?.setPitch(pitch.coerceIn(0.5f, 2.0f)) }

    val isSpeaking: Boolean get() = tts?.isSpeaking == true

    fun destroy() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isReady = false
    }
}
