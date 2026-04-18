package com.jarvis.app.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext

class LlamaEngine {

    private var modelHandle: Long = 0L
    private var modelPath: String = ""

    companion object {
        private const val TAG = "LlamaEngine"

        init {
            System.loadLibrary("jarvisai")
        }

        // Common prompt templates
        fun buildChatMLPrompt(systemPrompt: String, history: List<Pair<String,String>>, userMessage: String): String {
            val sb = StringBuilder()
            if (systemPrompt.isNotBlank()) {
                sb.append("<|im_start|>system\n$systemPrompt<|im_end|>\n")
            }
            for ((role, content) in history) {
                sb.append("<|im_start|>$role\n$content<|im_end|>\n")
            }
            sb.append("<|im_start|>user\n$userMessage<|im_end|>\n")
            sb.append("<|im_start|>assistant\n")
            return sb.toString()
        }

        fun buildLlamaPrompt(systemPrompt: String, userMessage: String): String =
            "[INST] <<SYS>>\n$systemPrompt\n<</SYS>>\n\n$userMessage [/INST]"
    }

    // JNI declarations
    private external fun nativeLoadModel(path: String, nThreads: Int, nCtx: Int): Long
    private external fun nativeGenerate(
        handle: Long, prompt: String, maxTokens: Int, temperature: Float, callback: TokenCallback
    ): Int
    private external fun nativeStopGeneration(handle: Long)
    private external fun nativeFreeModel(handle: Long)
    private external fun nativeGetModelDescription(handle: Long): String

    // ── Public API ───────────────────────────────────────────────────────────

    val isLoaded: Boolean get() = modelHandle != 0L
    val currentModelPath: String get() = modelPath

    suspend fun loadModel(path: String, nThreads: Int = 4, nCtx: Int = 4096): Boolean =
        withContext(Dispatchers.IO) {
            if (isLoaded) freeModel()
            modelHandle = nativeLoadModel(path, nThreads, nCtx)
            if (isLoaded) {
                modelPath = path
                Log.i(TAG, "Loaded: ${getModelDescription()}")
            } else {
                Log.e(TAG, "Failed to load $path")
            }
            isLoaded
        }

    fun generate(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f
    ): Flow<String> = callbackFlow {
        if (!isLoaded) {
            close(IllegalStateException("No model loaded"))
            return@callbackFlow
        }
        val callback = object : TokenCallback {
            override fun onToken(token: String): Boolean {
                trySend(token)
                return !isClosedForSend
            }
        }
        withContext(Dispatchers.IO) {
            nativeGenerate(modelHandle, prompt, maxTokens, temperature, callback)
        }
        close()
        awaitClose()
    }

    fun stopGeneration() {
        if (isLoaded) nativeStopGeneration(modelHandle)
    }

    fun getModelDescription(): String =
        if (isLoaded) nativeGetModelDescription(modelHandle) else "(no model)"

    fun freeModel() {
        if (isLoaded) {
            nativeFreeModel(modelHandle)
            modelHandle = 0L
            modelPath = ""
        }
    }

    // Callback interface called from native
    interface TokenCallback {
        fun onToken(token: String): Boolean
    }
}
