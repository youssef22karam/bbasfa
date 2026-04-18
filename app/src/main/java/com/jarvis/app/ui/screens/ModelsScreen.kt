package com.jarvis.app.ui.screens

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.*
import androidx.core.content.FileProvider
import com.jarvis.app.data.LocalModel
import com.jarvis.app.ui.theme.*
import com.jarvis.app.viewmodel.MainViewModel
import java.io.File

@Composable
fun ModelsScreen(vm: MainViewModel) {
    val localModels by vm.localModels.collectAsState()
    val activeModel by vm.activeModelName.collectAsState()
    val context     = LocalContext.current
    var showWhisper by remember { mutableStateOf(false) }

    // Import launcher
    val importLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { vm.importModel(it, isWhisper = showWhisper) }
    }

    LaunchedEffect(Unit) { vm.refreshLocalModels() }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(JarvisBg)
    ) {
        // Header
        Surface(color = JarvisSurface, modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(20.dp)) {
                Text("Local Models", style = MaterialTheme.typography.headlineMedium, color = JarvisBlue)
                Text(
                    "${localModels.count { !it.isWhisper }} LLM · ${localModels.count { it.isWhisper }} Whisper",
                    style = MaterialTheme.typography.bodyMedium,
                    color = JarvisTextMuted
                )
            }
        }

        // Import buttons
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Import LLM
            Button(
                onClick = { showWhisper = false; importLauncher.launch("*/*") },
                modifier = Modifier.weight(1f).height(56.dp),
                colors = ButtonDefaults.buttonColors(containerColor = JarvisBlue.copy(0.15f)),
                border = BorderStroke(1.dp, JarvisBlue.copy(0.6f)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.FileUpload, null, tint = JarvisBlue, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text("Import LLM", color = JarvisBlue)
            }
            // Import Whisper
            Button(
                onClick = { showWhisper = true; importLauncher.launch("*/*") },
                modifier = Modifier.weight(1f).height(56.dp),
                colors = ButtonDefaults.buttonColors(containerColor = JarvisGold.copy(0.1f)),
                border = BorderStroke(1.dp, JarvisGold.copy(0.6f)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.Mic, null, tint = JarvisGold, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text("Import Whisper", color = JarvisGold)
            }
        }

        // Refresh
        TextButton(
            onClick = { vm.refreshLocalModels() },
            modifier = Modifier.align(Alignment.End).padding(end = 16.dp)
        ) {
            Icon(Icons.Default.Refresh, null, tint = JarvisTextMuted, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(4.dp))
            Text("Refresh", color = JarvisTextMuted, style = MaterialTheme.typography.labelMedium)
        }

        if (localModels.isEmpty()) {
            EmptyModelsHint()
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                // LLMs
                val llms = localModels.filter { !it.isWhisper }
                if (llms.isNotEmpty()) {
                    item {
                        SectionLabel("Language Models")
                    }
                    items(llms, key = { it.id }) { model ->
                        ModelCard(
                            model = model,
                            isActive = model.name == activeModel,
                            onLoad   = { vm.loadModel(model.path) },
                            onDelete = { vm.deleteModel(model) },
                            onExport = { exportModel(context, model) }
                        )
                    }
                }
                // Whisper
                val whispers = localModels.filter { it.isWhisper }
                if (whispers.isNotEmpty()) {
                    item { SectionLabel("Whisper STT Models") }
                    items(whispers, key = { it.id }) { model ->
                        ModelCard(
                            model = model,
                            isActive = false,
                            isWhisper = true,
                            onLoad   = { vm.loadWhisperModel(model.path) },
                            onDelete = { vm.deleteModel(model) },
                            onExport = { exportModel(context, model) }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelCard(
    model: LocalModel,
    isActive: Boolean,
    isWhisper: Boolean = false,
    onLoad: () -> Unit,
    onDelete: () -> Unit,
    onExport: () -> Unit
) {
    val accentColor = if (isWhisper) JarvisGold else JarvisBlue
    var showConfirmDelete by remember { mutableStateOf(false) }

    Surface(
        color = if (isActive) accentColor.copy(0.1f) else JarvisSurface,
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(
            if (isActive) 1.5.dp else 1.dp,
            if (isActive) accentColor else JarvisBlueDark.copy(0.3f)
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                // Icon
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .background(accentColor.copy(0.15f), RoundedCornerShape(10.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        if (isWhisper) Icons.Default.RecordVoiceOver else Icons.Default.Psychology,
                        contentDescription = null, tint = accentColor, modifier = Modifier.size(24.dp)
                    )
                }
                Spacer(Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        model.name.take(28),
                        style = MaterialTheme.typography.titleMedium,
                        color = JarvisText
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Chip(model.sizeDisplay, JarvisTextMuted)
                        if (model.quantization != "?")
                            Chip(model.quantization, accentColor.copy(0.8f))
                        if (isActive)
                            Chip("ACTIVE", Color(0xFF00FF88))
                    }
                }
            }

            Spacer(Modifier.height(12.dp))

            // Action buttons
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                // Load button
                Button(
                    onClick = onLoad,
                    modifier = Modifier.weight(1f).height(44.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = accentColor),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(Icons.Default.PlayArrow, null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(4.dp))
                    Text(if (isWhisper) "Set STT" else "Load")
                }
                // Export
                OutlinedButton(
                    onClick = onExport,
                    modifier = Modifier.height(44.dp),
                    border = BorderStroke(1.dp, JarvisTextMuted.copy(0.5f)),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(Icons.Default.FileDownload, null, tint = JarvisTextMuted, modifier = Modifier.size(18.dp))
                }
                // Delete
                OutlinedButton(
                    onClick = { showConfirmDelete = true },
                    modifier = Modifier.height(44.dp),
                    border = BorderStroke(1.dp, JarvisRed.copy(0.5f)),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(Icons.Default.DeleteOutline, null, tint = JarvisRed, modifier = Modifier.size(18.dp))
                }
            }
        }
    }

    if (showConfirmDelete) {
        AlertDialog(
            onDismissRequest = { showConfirmDelete = false },
            containerColor   = JarvisSurface2,
            title = { Text("Delete model?", color = JarvisText) },
            text  = { Text("This will permanently delete ${model.name}", color = JarvisTextMuted) },
            confirmButton = {
                TextButton(onClick = { onDelete(); showConfirmDelete = false }) {
                    Text("Delete", color = JarvisRed)
                }
            },
            dismissButton = {
                TextButton(onClick = { showConfirmDelete = false }) {
                    Text("Cancel", color = JarvisTextMuted)
                }
            }
        )
    }
}

@Composable
private fun Chip(text: String, color: Color) {
    Surface(
        color = color.copy(0.12f),
        shape = RoundedCornerShape(6.dp)
    ) {
        Text(
            text,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
            style = MaterialTheme.typography.labelMedium,
            color = color
        )
    }
}

@Composable
private fun SectionLabel(label: String) {
    Text(
        label,
        style = MaterialTheme.typography.labelLarge,
        color = JarvisTextMuted,
        fontFamily = FontFamily.Monospace,
        modifier = Modifier.padding(vertical = 4.dp)
    )
}

@Composable
private fun EmptyModelsHint() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(Icons.Default.Inbox, null, tint = JarvisTextMuted, modifier = Modifier.size(64.dp))
            Spacer(Modifier.height(12.dp))
            Text("No models yet", color = JarvisText, style = MaterialTheme.typography.titleMedium)
            Text("Import a .gguf file or download from\nthe Download tab",
                color = JarvisTextMuted, style = MaterialTheme.typography.bodyMedium,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                modifier = Modifier.padding(top = 8.dp))
        }
    }
}

private fun exportModel(context: android.content.Context, model: LocalModel) {
    try {
        val file = File(model.path)
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "*/*"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(intent, "Export ${model.name}"))
    } catch (e: Exception) {
        android.util.Log.e("ModelsScreen", "Export failed", e)
    }
}
