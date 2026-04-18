package com.jarvis.app.repository

import android.content.Context
import android.util.Log
import com.jarvis.app.data.DownloadProgress
import com.jarvis.app.data.HFModel
import com.jarvis.app.data.HFModelFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

class HuggingFaceRepository(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    companion object {
        private const val TAG = "HuggingFaceRepo"
        private const val API_BASE = "https://huggingface.co/api"
        private const val HF_BASE = "https://huggingface.co"
        // Popular small GGUF models
        val FEATURED_MODELS = listOf(
            HFModel("Qwen/Qwen2.5-0.5B-Instruct-GGUF",     "Qwen 2.5 0.5B",   downloads=50_000),
            HFModel("Qwen/Qwen2.5-1.5B-Instruct-GGUF",     "Qwen 2.5 1.5B",   downloads=80_000),
            HFModel("microsoft/Phi-3.5-mini-instruct-gguf", "Phi-3.5 Mini 3.8B", downloads=120_000),
            HFModel("bartowski/gemma-2-2b-it-GGUF",         "Gemma-2 2B",      downloads=95_000),
            HFModel("lmstudio-community/Meta-Llama-3.2-1B-Instruct-GGUF", "Llama 3.2 1B", downloads=200_000),
            HFModel("lmstudio-community/Meta-Llama-3.2-3B-Instruct-GGUF", "Llama 3.2 3B", downloads=300_000),
            HFModel("ggerganov/whisper.cpp",                "Whisper Models",  downloads=500_000),
        )
    }

    suspend fun searchModels(query: String): List<HFModel> = withContext(Dispatchers.IO) {
        try {
            val url = "$API_BASE/models?filter=gguf&sort=downloads&direction=-1&limit=20&search=$query"
            val req = Request.Builder().url(url).build()
            val resp = client.newCall(req).execute()
            val body = resp.body?.string() ?: return@withContext emptyList()
            parseModelList(body)
        } catch (e: Exception) {
            Log.e(TAG, "Search failed", e)
            FEATURED_MODELS.filter { it.name.contains(query, ignoreCase = true) }
        }
    }

    suspend fun getModelFiles(modelId: String): List<HFModelFile> = withContext(Dispatchers.IO) {
        try {
            val url = "$API_BASE/models/$modelId"
            val req = Request.Builder().url(url).build()
            val resp = client.newCall(req).execute()
            val body = resp.body?.string() ?: return@withContext emptyList()
            parseModelFiles(body, modelId)
        } catch (e: Exception) {
            Log.e(TAG, "Get files failed for $modelId", e)
            emptyList()
        }
    }

    fun downloadModel(
        modelId: String,
        filename: String,
        destFile: File
    ): Flow<DownloadProgress> = flow {
        val url = "$HF_BASE/$modelId/resolve/main/$filename"
        Log.i(TAG, "Downloading $url → ${destFile.path}")
        try {
            val req = Request.Builder().url(url).build()
            val resp = client.newCall(req).execute()
            if (!resp.isSuccessful) throw Exception("HTTP ${resp.code}")

            val total = resp.body?.contentLength() ?: -1L
            val stream = resp.body?.byteStream() ?: throw Exception("Empty body")

            destFile.parentFile?.mkdirs()
            val buf = ByteArray(8192)
            var downloaded = 0L
            var lastEmit = 0L
            val startTime = System.currentTimeMillis()

            destFile.outputStream().use { out ->
                while (true) {
                    val n = stream.read(buf)
                    if (n == -1) break
                    out.write(buf, 0, n)
                    downloaded += n
                    if (downloaded - lastEmit > 512_000 || downloaded == total) {
                        val elapsed = (System.currentTimeMillis() - startTime) / 1000f
                        val speed = if (elapsed > 0) "%.1f MB/s".format(downloaded / 1024f / 1024f / elapsed) else ""
                        emit(DownloadProgress(modelId, filename, downloaded, total, speed))
                        lastEmit = downloaded
                    }
                }
            }
            emit(DownloadProgress(modelId, filename, downloaded, downloaded, "Done"))
        } catch (e: Exception) {
            Log.e(TAG, "Download failed", e)
            destFile.delete()
            throw e
        }
    }

    fun getModelsDir(): File = File(context.filesDir, "models").also { it.mkdirs() }
    fun getWhisperDir(): File = File(context.filesDir, "whisper").also { it.mkdirs() }

    // ── Parsing ──────────────────────────────────────────────────────────────

    private fun parseModelList(json: String): List<HFModel> {
        val arr = JSONArray(json)
        return (0 until minOf(arr.length(), 20)).mapNotNull {
            runCatching {
                val obj = arr.getJSONObject(it)
                HFModel(
                    modelId = obj.getString("modelId"),
                    name = obj.getString("modelId").substringAfterLast('/'),
                    downloads = obj.optLong("downloads", 0),
                    likes = obj.optLong("likes", 0)
                )
            }.getOrNull()
        }
    }

    private fun parseModelFiles(json: String, modelId: String): List<HFModelFile> {
        val obj = JSONObject(json)
        val siblings = obj.optJSONArray("siblings") ?: return emptyList()
        return (0 until siblings.length()).mapNotNull {
            runCatching {
                val f = siblings.getJSONObject(it)
                val name = f.getString("rfilename")
                if (!name.endsWith(".gguf", ignoreCase = true) &&
                    !name.endsWith(".bin", ignoreCase = true)) return@runCatching null
                val size = f.optLong("size", 0L)
                val quant = extractQuantization(name)
                HFModelFile(
                    filename = name,
                    sizeBytes = size,
                    downloadUrl = "$HF_BASE/$modelId/resolve/main/$name",
                    quantization = quant
                )
            }.getOrNull()
        }.filterNotNull().sortedWith(
            compareByDescending<HFModelFile> { it.quantization == "Q4_K_M" }
                .thenBy { it.sizeBytes }
        )
    }

    private fun extractQuantization(filename: String): String {
        val upper = filename.uppercase()
        return when {
            "Q8_0" in upper   -> "Q8_0"
            "Q6_K" in upper   -> "Q6_K"
            "Q5_K_M" in upper -> "Q5_K_M"
            "Q5_K_S" in upper -> "Q5_K_S"
            "Q4_K_M" in upper -> "Q4_K_M"
            "Q4_K_S" in upper -> "Q4_K_S"
            "Q4_0" in upper   -> "Q4_0"
            "Q3_K_M" in upper -> "Q3_K_M"
            "Q2_K" in upper   -> "Q2_K"
            "IQ4_NL" in upper -> "IQ4_NL"
            "IQ3_M" in upper  -> "IQ3_M"
            "F16" in upper    -> "F16"
            else              -> "unknown"
        }
    }
}
