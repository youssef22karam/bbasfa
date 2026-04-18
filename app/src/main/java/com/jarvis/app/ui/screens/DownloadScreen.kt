package com.jarvis.app.ui.screens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.*
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.*
import com.jarvis.app.data.*
import com.jarvis.app.ui.theme.*
import com.jarvis.app.viewmodel.MainViewModel

@Composable
fun DownloadScreen(vm: MainViewModel) {
    val hfResults     by vm.hfResults.collectAsState()
    val modelFiles    by vm.modelFiles.collectAsState()
    val downloadProg  by vm.downloadProgress.collectAsState()
    var query         by remember { mutableStateOf("") }
    var selectedModel by remember { mutableStateOf<HFModel?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(JarvisBg)
    ) {
        // Header
        Surface(color = JarvisSurface, modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(20.dp)) {
                Text("Download Models", style = MaterialTheme.typography.headlineMedium, color = JarvisBlue)
                Text("Search HuggingFace for GGUF models", style = MaterialTheme.typography.bodyMedium, color = JarvisTextMuted)
            }
        }

        // Download progress bar
        downloadProg?.let { prog ->
            DownloadProgressBar(prog)
        }

        // Search bar
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Search models (e.g. qwen, llama, gemma)…", color = JarvisTextMuted) },
                leadingIcon = { Icon(Icons.Default.Search, null, tint = JarvisBlue) },
                shape = RoundedCornerShape(14.dp),
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                keyboardActions = KeyboardActions(onSearch = { vm.searchHuggingFace(query) }),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor   = JarvisBlue,
                    unfocusedBorderColor = JarvisBlueDark.copy(0.4f),
                    focusedTextColor     = JarvisText,
                    unfocusedTextColor   = JarvisText,
                    cursorColor          = JarvisBlue,
                    focusedContainerColor   = JarvisSurface,
                    unfocusedContainerColor = JarvisSurface
                )
            )
            Button(
                onClick = { vm.searchHuggingFace(query) },
                modifier = Modifier.height(56.dp),
                colors = ButtonDefaults.buttonColors(containerColor = JarvisBlue),
                shape = RoundedCornerShape(14.dp)
            ) {
                Icon(Icons.Default.Search, null)
            }
        }

        if (selectedModel == null) {
            // Model list
            Text(
                if (query.isBlank()) "⬡ FEATURED MODELS" else "SEARCH RESULTS",
                style = MaterialTheme.typography.labelLarge,
                color = JarvisTextMuted,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
            )
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(hfResults, key = { it.modelId }) { model ->
                    HFModelCard(model = model, onClick = {
                        selectedModel = model
                        vm.loadModelFiles(model.modelId)
                    })
                }
            }
        } else {
            // File list for selected model
            Surface(
                color = JarvisSurface,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(onClick = { selectedModel = null }) {
                        Icon(Icons.Default.ArrowBack, null, tint = JarvisBlue)
                    }
                    Column {
                        Text(
                            selectedModel!!.name,
                            style = MaterialTheme.typography.titleMedium,
                            color = JarvisText
                        )
                        Text(
                            selectedModel!!.modelId,
                            style = MaterialTheme.typography.labelSmall,
                            color = JarvisTextMuted,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }

            val isWhisper = "whisper" in selectedModel!!.modelId.lowercase()

            if (modelFiles.isEmpty()) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = JarvisBlue)
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    item {
                        if (isWhisper) {
                            InfoBanner("💡 Whisper models will be used for voice-to-text")
                        } else {
                            InfoBanner("💡 Q4_K_M recommended — best quality/size balance for mobile")
                        }
                    }
                    items(modelFiles, key = { it.filename }) { file ->
                        ModelFileCard(
                            file = file,
                            isDownloading = downloadProg?.filename == file.filename,
                            progress = if (downloadProg?.filename == file.filename) downloadProg else null,
                            onDownload = {
                                vm.downloadModel(selectedModel!!.modelId, file, isWhisper)
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun HFModelCard(model: HFModel, onClick: () -> Unit) {
    Surface(
        color = JarvisSurface,
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(1.dp, JarvisBlueDark.copy(0.3f)),
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .background(JarvisBlue.copy(0.12f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Hub, null, tint = JarvisBlue, modifier = Modifier.size(28.dp))
            }
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(model.name, style = MaterialTheme.typography.titleMedium, color = JarvisText,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(model.modelId, style = MaterialTheme.typography.labelSmall, color = JarvisTextMuted,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                if (model.downloads > 0) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 4.dp)) {
                        Icon(Icons.Default.Download, null, tint = JarvisTextMuted, modifier = Modifier.size(12.dp))
                        Text("${formatCount(model.downloads)}", style = MaterialTheme.typography.labelSmall, color = JarvisTextMuted)
                        Icon(Icons.Default.Favorite, null, tint = JarvisRed.copy(0.7f), modifier = Modifier.size(12.dp))
                        Text("${formatCount(model.likes)}", style = MaterialTheme.typography.labelSmall, color = JarvisTextMuted)
                    }
                }
            }
            Icon(Icons.Default.ChevronRight, null, tint = JarvisTextMuted)
        }
    }
}

@Composable
private fun ModelFileCard(
    file: HFModelFile,
    isDownloading: Boolean,
    progress: DownloadProgress?,
    onDownload: () -> Unit
) {
    val isRecommended = file.quantization == "Q4_K_M"
    Surface(
        color = if (isRecommended) JarvisBlue.copy(0.06f) else JarvisSurface,
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(
            if (isRecommended) 1.5.dp else 1.dp,
            if (isRecommended) JarvisBlue.copy(0.5f) else JarvisBlueDark.copy(0.25f)
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            file.filename.take(32),
                            style = MaterialTheme.typography.titleMedium,
                            color = JarvisText,
                            maxLines = 2
                        )
                        if (isRecommended) {
                            Surface(color = JarvisBlue.copy(0.2f), shape = RoundedCornerShape(6.dp)) {
                                Text("★ Best", modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                    style = MaterialTheme.typography.labelSmall, color = JarvisBlue)
                            }
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 4.dp)) {
                        QuantChip(file.quantization)
                        Surface(color = JarvisTextMuted.copy(0.1f), shape = RoundedCornerShape(6.dp)) {
                            Text(file.sizeDisplay, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                style = MaterialTheme.typography.labelSmall, color = JarvisTextMuted)
                        }
                    }
                }
                // Download button
                FilledIconButton(
                    onClick = onDownload,
                    enabled = !isDownloading,
                    modifier = Modifier.size(50.dp),
                    colors = IconButtonDefaults.filledIconButtonColors(containerColor = JarvisBlue),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(Icons.Default.Download, "Download", tint = Color(0xFF001F29))
                }
            }

            // Progress bar
            if (isDownloading && progress != null) {
                Spacer(Modifier.height(10.dp))
                LinearProgressIndicator(
                    progress = { progress.percent / 100f },
                    modifier = Modifier.fillMaxWidth(),
                    color = JarvisBlue,
                    trackColor = JarvisBlueDark.copy(0.2f)
                )
                Row(
                    modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("${progress.percent}%", style = MaterialTheme.typography.labelSmall, color = JarvisBlue)
                    Text(progress.speed, style = MaterialTheme.typography.labelSmall, color = JarvisTextMuted)
                }
            }
        }
    }
}

@Composable
private fun QuantChip(quant: String) {
    val color = when (quant) {
        "Q4_K_M", "Q5_K_M" -> JarvisBlue
        "Q8_0"              -> Color(0xFF00FF88)
        "Q3_K_M", "Q2_K"   -> JarvisRed
        "F16"               -> JarvisGold
        else                -> JarvisTextMuted
    }
    Surface(color = color.copy(0.12f), shape = RoundedCornerShape(6.dp)) {
        Text(quant, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
            style = MaterialTheme.typography.labelSmall, color = color)
    }
}

@Composable
private fun DownloadProgressBar(prog: DownloadProgress) {
    Surface(color = JarvisSurface2, modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Downloading: ${prog.filename.take(28)}", style = MaterialTheme.typography.labelMedium, color = JarvisBlue)
                Text("${prog.percent}%  ${prog.speed}", style = MaterialTheme.typography.labelSmall, color = JarvisTextMuted)
            }
            Spacer(Modifier.height(6.dp))
            LinearProgressIndicator(
                progress = { prog.percent / 100f },
                modifier = Modifier.fillMaxWidth(),
                color = JarvisBlue,
                trackColor = JarvisBlueDark.copy(0.2f)
            )
        }
    }
}

@Composable
private fun InfoBanner(text: String) {
    Surface(
        color = JarvisBlue.copy(0.08f),
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, JarvisBlue.copy(0.2f))
    ) {
        Text(text, modifier = Modifier.padding(12.dp),
            style = MaterialTheme.typography.bodySmall, color = JarvisTextMuted)
    }
}

private fun formatCount(n: Long): String = when {
    n >= 1_000_000 -> "%.1fM".format(n / 1_000_000f)
    n >= 1_000     -> "%.0fK".format(n / 1_000f)
    else           -> n.toString()
}
