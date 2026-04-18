package com.jarvis.app.data

import kotlinx.serialization.Serializable

// ─────────────────────────────────────────────────────────────────────────────
// Local model metadata (stored on device)
// ─────────────────────────────────────────────────────────────────────────────
@Serializable
data class LocalModel(
    val id: String,
    val name: String,
    val path: String,
    val sizeBytes: Long,
    val quantization: String = "unknown",
    val isWhisper: Boolean = false,
    val addedAt: Long = System.currentTimeMillis()
) {
    val sizeMB: Float get() = sizeBytes / (1024f * 1024f)
    val sizeDisplay: String get() = when {
        sizeBytes > 1024 * 1024 * 1024 -> "%.1f GB".format(sizeBytes / (1024f * 1024f * 1024f))
        else -> "%.0f MB".format(sizeMB)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HuggingFace API models
// ─────────────────────────────────────────────────────────────────────────────
@Serializable
data class HFModel(
    val modelId: String,
    val name: String,
    val downloads: Long = 0,
    val likes: Long = 0,
    val tags: List<String> = emptyList(),
    val lastModified: String = ""
)

@Serializable
data class HFModelFile(
    val filename: String,
    val sizeBytes: Long,
    val downloadUrl: String,
    val quantization: String
) {
    val sizeDisplay: String get() = when {
        sizeBytes > 1024 * 1024 * 1024 -> "%.1f GB".format(sizeBytes / (1024f * 1024f * 1024f))
        else -> "%.0f MB".format(sizeBytes / (1024f * 1024f))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat
// ─────────────────────────────────────────────────────────────────────────────
data class ChatMessage(
    val id: String = java.util.UUID.randomUUID().toString(),
    val role: ChatRole,
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val isStreaming: Boolean = false
)

enum class ChatRole { USER, ASSISTANT, SYSTEM }

// ─────────────────────────────────────────────────────────────────────────────
// App state
// ─────────────────────────────────────────────────────────────────────────────
enum class JarvisState {
    IDLE, LISTENING, TRANSCRIBING, THINKING, SPEAKING, ERROR
}

data class DownloadProgress(
    val modelId: String,
    val filename: String,
    val downloaded: Long,
    val total: Long,
    val speed: String = ""
) {
    val percent: Int get() = if (total > 0) ((downloaded * 100) / total).toInt() else 0
    val isComplete: Boolean get() = downloaded >= total && total > 0
}
