package com.jarvis.app.engine

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WhisperEngine {

    private var contextPtr: Long = 0L

    companion object {
        private const val TAG = "WhisperEngine"
        private const val SAMPLE_RATE = 16_000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

        init {
            System.loadLibrary("jarvisai")
        }
    }

    // JNI declarations
    private external fun nativeInitContext(modelPath: String): Long
    private external fun nativeFreeContext(ctxPtr: Long)
    private external fun nativeTranscribe(ctxPtr: Long, pcmFloat: FloatArray, language: String): String

    // ── Recording state ──────────────────────────────────────────────────────

    private var audioRecord: AudioRecord? = null
    @Volatile private var isRecording = false
    private val recordBuffer = mutableListOf<Float>()

    val isLoaded: Boolean get() = contextPtr != 0L

    // ── Public API ───────────────────────────────────────────────────────────

    suspend fun loadModel(path: String): Boolean = withContext(Dispatchers.IO) {
        if (isLoaded) freeModel()
        contextPtr = nativeInitContext(path)
        if (isLoaded) Log.i(TAG, "Whisper loaded: $path")
        else          Log.e(TAG, "Whisper load failed: $path")
        isLoaded
    }

    fun startRecording() {
        if (isRecording) return
        val bufSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE, CHANNEL, ENCODING, bufSize
        )
        recordBuffer.clear()
        isRecording = true
        audioRecord?.startRecording()

        Thread {
            val buf = ShortArray(bufSize / 2)
            while (isRecording) {
                val read = audioRecord?.read(buf, 0, buf.size) ?: 0
                if (read > 0) {
                    for (i in 0 until read) {
                        recordBuffer.add(buf[i].toFloat() / 32768f)
                    }
                }
            }
        }.start()
    }

    suspend fun stopAndTranscribe(language: String = "en"): String = withContext(Dispatchers.IO) {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        if (!isLoaded) {
            return@withContext "[Whisper not loaded]"
        }

        val samples = recordBuffer.toFloatArray()
        if (samples.isEmpty()) return@withContext ""

        Log.i(TAG, "Transcribing ${samples.size} samples (${samples.size / SAMPLE_RATE}s)")
        val result = nativeTranscribe(contextPtr, samples, language)
        recordBuffer.clear()
        result
    }

    fun freeModel() {
        isRecording = false
        audioRecord?.release()
        audioRecord = null
        if (isLoaded) {
            nativeFreeContext(contextPtr)
            contextPtr = 0L
        }
    }
}
