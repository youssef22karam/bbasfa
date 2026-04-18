package com.jarvis.app.viewmodel

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.jarvis.app.data.*
import com.jarvis.app.engine.LlamaEngine
import com.jarvis.app.engine.TTSEngine
import com.jarvis.app.engine.WhisperEngine
import com.jarvis.app.repository.HuggingFaceRepository
import com.jarvis.app.service.JarvisAccessibilityService
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs   = AppPreferences(app)
    private val hfRepo  = HuggingFaceRepository(app)
    val llamaEngine     = LlamaEngine()
    val whisperEngine   = WhisperEngine()
    val ttsEngine       = TTSEngine(app)

    // ── UI State ─────────────────────────────────────────────────────────────

    private val _jarvisState = MutableStateFlow(JarvisState.IDLE)
    val jarvisState: StateFlow<JarvisState> = _jarvisState

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages

    private val _localModels = MutableStateFlow<List<LocalModel>>(emptyList())
    val localModels: StateFlow<List<LocalModel>> = _localModels

    private val _hfResults = MutableStateFlow<List<com.jarvis.app.data.HFModel>>(emptyList())
    val hfResults: StateFlow<List<com.jarvis.app.data.HFModel>> = _hfResults

    private val _modelFiles = MutableStateFlow<List<com.jarvis.app.data.HFModelFile>>(emptyList())
    val modelFiles: StateFlow<List<com.jarvis.app.data.HFModelFile>> = _modelFiles

    private val _downloadProgress = MutableStateFlow<DownloadProgress?>(null)
    val downloadProgress: StateFlow<DownloadProgress?> = _downloadProgress

    private val _activeModelName = MutableStateFlow("No model loaded")
    val activeModelName: StateFlow<String> = _activeModelName

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error

    private val _statusText = MutableStateFlow("Ready")
    val statusText: StateFlow<String> = _statusText

    // Inference settings (live from prefs)
    val systemPrompt = prefs.systemPrompt.stateIn(viewModelScope, SharingStarted.Eagerly, AppPreferences.DEFAULT_SYSTEM_PROMPT)
    val maxTokens    = prefs.maxTokens.stateIn(viewModelScope, SharingStarted.Eagerly, AppPreferences.DEFAULT_MAX_TOKENS)
    val temperature  = prefs.temperature.stateIn(viewModelScope, SharingStarted.Eagerly, AppPreferences.DEFAULT_TEMPERATURE)
    val nThreads     = prefs.nThreads.stateIn(viewModelScope, SharingStarted.Eagerly, AppPreferences.DEFAULT_N_THREADS)
    val ttsRate      = prefs.ttsRate.stateIn(viewModelScope, SharingStarted.Eagerly, AppPreferences.DEFAULT_TTS_RATE)
    val ttsPitch     = prefs.ttsPitch.stateIn(viewModelScope, SharingStarted.Eagerly, AppPreferences.DEFAULT_TTS_PITCH)

    private var generationJob: Job? = null
    private val conversationHistory = mutableListOf<Pair<String, String>>()

    companion object {
        private const val TAG = "MainViewModel"
    }

    // ── Initialization ───────────────────────────────────────────────────────

    init {
        viewModelScope.launch {
            ttsEngine.init()
            refreshLocalModels()
            // Auto-load last used model
            prefs.activeModelPath.first().takeIf { it.isNotBlank() }?.let { path ->
                if (File(path).exists()) loadModel(path)
            }
            // Load featured models by default
            _hfResults.value = HuggingFaceRepository.FEATURED_MODELS
        }
    }

    // ── Voice interaction ────────────────────────────────────────────────────

    fun startListening() {
        if (_jarvisState.value != JarvisState.IDLE) return
        _jarvisState.value = JarvisState.LISTENING
        _statusText.value = "Listening…"
        whisperEngine.startRecording()
    }

    fun stopListeningAndProcess() {
        if (_jarvisState.value != JarvisState.LISTENING) return
        viewModelScope.launch {
            _jarvisState.value = JarvisState.TRANSCRIBING
            _statusText.value = "Transcribing…"
            val transcript = whisperEngine.stopAndTranscribe(prefs.language.first())
            if (transcript.isNotBlank()) {
                sendMessage(transcript)
            } else {
                _jarvisState.value = JarvisState.IDLE
                _statusText.value = "Ready"
            }
        }
    }

    // ── Chat ─────────────────────────────────────────────────────────────────

    fun sendMessage(text: String) {
        if (text.isBlank()) return
        if (!llamaEngine.isLoaded) {
            _error.value = "No LLM loaded. Please select a model first."
            return
        }

        val userMsg = ChatMessage(role = ChatRole.USER, content = text)
        _messages.update { it + userMsg }

        val assistantId = java.util.UUID.randomUUID().toString()
        val placeholder = ChatMessage(id = assistantId, role = ChatRole.ASSISTANT, content = "", isStreaming = true)
        _messages.update { it + placeholder }

        _jarvisState.value = JarvisState.THINKING
        _statusText.value = "Thinking…"

        generationJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val prompt = LlamaEngine.buildChatMLPrompt(
                    systemPrompt.value,
                    conversationHistory.takeLast(10),
                    text
                )
                val sb = StringBuilder()

                llamaEngine.generate(prompt, maxTokens.value, temperature.value)
                    .collect { token ->
                        sb.append(token)
                        _messages.update { list ->
                            list.map { if (it.id == assistantId) it.copy(content = sb.toString()) else it }
                        }
                    }

                val response = sb.toString().trim()
                // Finalise message
                _messages.update { list ->
                    list.map { if (it.id == assistantId) it.copy(content = response, isStreaming = false) else it }
                }
                conversationHistory.add("user" to text)
                conversationHistory.add("assistant" to response)

                // TTS
                withContext(Dispatchers.Main) {
                    _jarvisState.value = JarvisState.SPEAKING
                    _statusText.value = "Speaking…"
                    ttsEngine.speak(response)
                }

                // Check for accessibility commands
                handleAccessibilityCommand(response)

            } catch (e: Exception) {
                Log.e(TAG, "Generation error", e)
                _error.value = "Generation failed: ${e.message}"
            } finally {
                withContext(Dispatchers.Main) {
                    _jarvisState.value = JarvisState.IDLE
                    _statusText.value = "Ready"
                }
            }
        }
    }

    fun stopGeneration() {
        generationJob?.cancel()
        llamaEngine.stopGeneration()
        ttsEngine.stop()
        _jarvisState.value = JarvisState.IDLE
        _statusText.value = "Stopped"
    }

    fun clearChat() {
        _messages.value = emptyList()
        conversationHistory.clear()
    }

    // ── Accessibility commands ────────────────────────────────────────────────

    private fun handleAccessibilityCommand(response: String) {
        val svc = JarvisAccessibilityService.instance ?: return
        val lower = response.lowercase()
        when {
            "click" in lower || "tap" in lower || "press" in lower -> {
                svc.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK)
            }
            "go back" in lower -> svc.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK)
            "home" in lower    -> svc.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_HOME)
            "recent" in lower  -> svc.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_RECENTS)
        }
    }

    fun executeAccessibilityAction(action: String) {
        JarvisAccessibilityService.instance?.executeAction(action)
    }

    // ── Model management ──────────────────────────────────────────────────────

    fun loadModel(path: String) {
        viewModelScope.launch {
            _statusText.value = "Loading model…"
            val ok = llamaEngine.loadModel(path, nThreads.value)
            if (ok) {
                _activeModelName.value = File(path).nameWithoutExtension
                prefs.setActiveModelPath(path)
                _statusText.value = "Model ready"
                ttsEngine.speak("Model loaded. I'm ready.")
            } else {
                _error.value = "Failed to load model: ${File(path).name}"
                _statusText.value = "Load failed"
            }
        }
    }

    fun loadWhisperModel(path: String) {
        viewModelScope.launch {
            val ok = whisperEngine.loadModel(path)
            if (ok) {
                prefs.setWhisperModelPath(path)
            } else {
                _error.value = "Failed to load Whisper model"
            }
        }
    }

    fun refreshLocalModels() {
        viewModelScope.launch(Dispatchers.IO) {
            val models = mutableListOf<LocalModel>()
            listOf(hfRepo.getModelsDir(), hfRepo.getWhisperDir()).forEach { dir ->
                dir.listFiles()?.filter { it.extension in listOf("gguf", "bin") }?.forEach { f ->
                    models.add(LocalModel(
                        id = f.nameWithoutExtension,
                        name = f.nameWithoutExtension,
                        path = f.absolutePath,
                        sizeBytes = f.length(),
                        isWhisper = f.parent?.contains("whisper") == true,
                        quantization = extractQuant(f.name)
                    ))
                }
            }
            _localModels.value = models
        }
    }

    private fun extractQuant(name: String): String {
        val u = name.uppercase()
        return listOf("Q4_K_M","Q5_K_M","Q8_0","Q4_0","Q6_K","Q3_K_M","F16").firstOrNull { it in u } ?: "?"
    }

    // ── HuggingFace ────────────────────────────────────────────────────────────

    fun searchHuggingFace(query: String) {
        viewModelScope.launch {
            _hfResults.value = if (query.isBlank()) HuggingFaceRepository.FEATURED_MODELS
            else hfRepo.searchModels(query)
        }
    }

    fun loadModelFiles(modelId: String) {
        viewModelScope.launch {
            _modelFiles.value = emptyList()
            _modelFiles.value = hfRepo.getModelFiles(modelId)
        }
    }

    fun downloadModel(modelId: String, file: com.jarvis.app.data.HFModelFile, isWhisper: Boolean = false) {
        val dir = if (isWhisper) hfRepo.getWhisperDir() else hfRepo.getModelsDir()
        val destFile = File(dir, file.filename)
        viewModelScope.launch(Dispatchers.IO) {
            hfRepo.downloadModel(modelId, file.filename, destFile)
                .collect { progress ->
                    _downloadProgress.value = progress
                    if (progress.isComplete) {
                        refreshLocalModels()
                        _downloadProgress.value = null
                    }
                }
        }
    }

    fun deleteModel(model: LocalModel) {
        viewModelScope.launch(Dispatchers.IO) {
            File(model.path).delete()
            refreshLocalModels()
        }
    }

    // ── Import / Export ────────────────────────────────────────────────────────

    fun importModel(sourceUri: Uri, isWhisper: Boolean = false) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val cr = getApplication<Application>().contentResolver
                val fileName = getFileName(cr, sourceUri) ?: "imported_${System.currentTimeMillis()}.gguf"
                val dir = if (isWhisper) hfRepo.getWhisperDir() else hfRepo.getModelsDir()
                val dest = File(dir, fileName)
                cr.openInputStream(sourceUri)?.use { input ->
                    dest.outputStream().use { out -> input.copyTo(out) }
                }
                refreshLocalModels()
                _statusText.value = "Imported: $fileName"
            } catch (e: Exception) {
                _error.value = "Import failed: ${e.message}"
            }
        }
    }

    private fun getFileName(cr: android.content.ContentResolver, uri: Uri): String? {
        var name: String? = null
        cr.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (idx != -1) name = cursor.getString(idx)
            }
        }
        return name
    }

    // ── Settings ────────────────────────────────────────────────────────────────

    fun updateSystemPrompt(p: String)  = viewModelScope.launch { prefs.setSystemPrompt(p) }
    fun updateMaxTokens(n: Int)        = viewModelScope.launch { prefs.setMaxTokens(n) }
    fun updateTemperature(t: Float)    = viewModelScope.launch { prefs.setTemperature(t) }
    fun updateNThreads(n: Int)         = viewModelScope.launch { prefs.setNThreads(n) }
    fun updateTtsRate(r: Float)        = viewModelScope.launch { prefs.setTtsRate(r); ttsEngine.setRate(r) }
    fun updateTtsPitch(p: Float)       = viewModelScope.launch { prefs.setTtsPitch(p); ttsEngine.setPitch(p) }

    fun clearError() { _error.value = null }

    override fun onCleared() {
        super.onCleared()
        llamaEngine.freeModel()
        whisperEngine.freeModel()
        ttsEngine.destroy()
    }
}
