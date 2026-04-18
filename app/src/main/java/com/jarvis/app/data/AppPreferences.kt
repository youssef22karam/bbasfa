package com.jarvis.app.data

import android.content.Context
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "jarvis_prefs")

class AppPreferences(private val context: Context) {

    companion object {
        val KEY_ACTIVE_MODEL_PATH  = stringPreferencesKey("active_model_path")
        val KEY_WHISPER_MODEL_PATH = stringPreferencesKey("whisper_model_path")
        val KEY_SYSTEM_PROMPT      = stringPreferencesKey("system_prompt")
        val KEY_MAX_TOKENS         = intPreferencesKey("max_tokens")
        val KEY_TEMPERATURE        = floatPreferencesKey("temperature")
        val KEY_N_THREADS          = intPreferencesKey("n_threads")
        val KEY_CTX_SIZE           = intPreferencesKey("ctx_size")
        val KEY_TTS_RATE           = floatPreferencesKey("tts_rate")
        val KEY_TTS_PITCH          = floatPreferencesKey("tts_pitch")
        val KEY_VOICE_ALWAYS_ON    = booleanPreferencesKey("voice_always_on")
        val KEY_LANGUAGE           = stringPreferencesKey("language")

        const val DEFAULT_SYSTEM_PROMPT = """You are JARVIS, an advanced AI assistant running locally on this Android device. You are helpful, concise, and capable. When asked to perform actions on the phone (like liking posts, writing notes, clicking buttons), describe what you're doing step by step. Keep responses brief and conversational unless asked for detail."""
        const val DEFAULT_MAX_TOKENS = 512
        const val DEFAULT_TEMPERATURE = 0.7f
        const val DEFAULT_N_THREADS = 4
        const val DEFAULT_CTX_SIZE = 4096
        const val DEFAULT_TTS_RATE = 1.0f
        const val DEFAULT_TTS_PITCH = 0.95f
    }

    val activeModelPath: Flow<String> = context.dataStore.data.map {
        it[KEY_ACTIVE_MODEL_PATH] ?: ""
    }
    val whisperModelPath: Flow<String> = context.dataStore.data.map {
        it[KEY_WHISPER_MODEL_PATH] ?: ""
    }
    val systemPrompt: Flow<String> = context.dataStore.data.map {
        it[KEY_SYSTEM_PROMPT] ?: DEFAULT_SYSTEM_PROMPT
    }
    val maxTokens: Flow<Int> = context.dataStore.data.map {
        it[KEY_MAX_TOKENS] ?: DEFAULT_MAX_TOKENS
    }
    val temperature: Flow<Float> = context.dataStore.data.map {
        it[KEY_TEMPERATURE] ?: DEFAULT_TEMPERATURE
    }
    val nThreads: Flow<Int> = context.dataStore.data.map {
        it[KEY_N_THREADS] ?: DEFAULT_N_THREADS
    }
    val ctxSize: Flow<Int> = context.dataStore.data.map {
        it[KEY_CTX_SIZE] ?: DEFAULT_CTX_SIZE
    }
    val ttsRate: Flow<Float> = context.dataStore.data.map {
        it[KEY_TTS_RATE] ?: DEFAULT_TTS_RATE
    }
    val ttsPitch: Flow<Float> = context.dataStore.data.map {
        it[KEY_TTS_PITCH] ?: DEFAULT_TTS_PITCH
    }
    val language: Flow<String> = context.dataStore.data.map {
        it[KEY_LANGUAGE] ?: "en"
    }

    suspend fun setActiveModelPath(path: String) =
        context.dataStore.edit { it[KEY_ACTIVE_MODEL_PATH] = path }

    suspend fun setWhisperModelPath(path: String) =
        context.dataStore.edit { it[KEY_WHISPER_MODEL_PATH] = path }

    suspend fun setSystemPrompt(prompt: String) =
        context.dataStore.edit { it[KEY_SYSTEM_PROMPT] = prompt }

    suspend fun setMaxTokens(n: Int) =
        context.dataStore.edit { it[KEY_MAX_TOKENS] = n }

    suspend fun setTemperature(t: Float) =
        context.dataStore.edit { it[KEY_TEMPERATURE] = t }

    suspend fun setNThreads(n: Int) =
        context.dataStore.edit { it[KEY_N_THREADS] = n }

    suspend fun setCtxSize(n: Int) =
        context.dataStore.edit { it[KEY_CTX_SIZE] = n }

    suspend fun setTtsRate(r: Float) =
        context.dataStore.edit { it[KEY_TTS_RATE] = r }

    suspend fun setTtsPitch(p: Float) =
        context.dataStore.edit { it[KEY_TTS_PITCH] = p }

    suspend fun setLanguage(lang: String) =
        context.dataStore.edit { it[KEY_LANGUAGE] = lang }
}
