package com.jarvis.app.ui.screens

import android.content.Intent
import android.provider.Settings
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.*
import com.jarvis.app.data.AppPreferences
import com.jarvis.app.ui.theme.*
import com.jarvis.app.viewmodel.MainViewModel
import kotlin.math.roundToInt

@Composable
fun SettingsScreen(vm: MainViewModel) {
    val context = LocalContext.current
    val systemPrompt by vm.systemPrompt.collectAsState()
    val maxTokens    by vm.maxTokens.collectAsState()
    val temperature  by vm.temperature.collectAsState()
    val nThreads     by vm.nThreads.collectAsState()
    val ttsRate      by vm.ttsRate.collectAsState()
    val ttsPitch     by vm.ttsPitch.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(JarvisBg)
            .verticalScroll(rememberScrollState())
    ) {
        // Header
        Surface(color = JarvisSurface, modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(20.dp)) {
                Text("Settings", style = MaterialTheme.typography.headlineMedium, color = JarvisBlue)
                Text("Configure JARVIS behaviour", style = MaterialTheme.typography.bodyMedium, color = JarvisTextMuted)
            }
        }

        Spacer(Modifier.height(12.dp))

        // ── Accessibility ─────────────────────────────────────────────────────
        SettingsSection("Phone Control") {
            ActionButton(
                icon  = Icons.Default.Accessibility,
                title = "Enable Accessibility Service",
                subtitle = "Required to control your phone, click buttons, type text",
                color = JarvisBlue,
                onClick = {
                    context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                }
            )
            ActionButton(
                icon  = Icons.Default.Screenshot,
                title = "Allow Overlay Permission",
                subtitle = "Required for on-screen overlay",
                color = JarvisGold,
                onClick = {
                    context.startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION))
                }
            )
        }

        // ── Model inference ────────────────────────────────────────────────────
        SettingsSection("Inference") {
            SliderSetting(
                title    = "Max Tokens",
                value    = maxTokens.toFloat(),
                range    = 64f..2048f,
                steps    = 15,
                display  = maxTokens.toString(),
                onChange = { vm.updateMaxTokens(it.roundToInt()) }
            )
            SliderSetting(
                title    = "Temperature",
                value    = temperature,
                range    = 0f..2f,
                steps    = 19,
                display  = "%.2f".format(temperature),
                onChange = { vm.updateTemperature(it) }
            )
            SliderSetting(
                title    = "CPU Threads",
                value    = nThreads.toFloat(),
                range    = 1f..8f,
                steps    = 6,
                display  = nThreads.toString(),
                onChange = { vm.updateNThreads(it.roundToInt()) }
            )
        }

        // ── Voice / TTS ────────────────────────────────────────────────────────
        SettingsSection("Voice") {
            SliderSetting(
                title    = "Speech Rate",
                value    = ttsRate,
                range    = 0.5f..2f,
                steps    = 14,
                display  = "%.1fx".format(ttsRate),
                onChange = { vm.updateTtsRate(it) }
            )
            SliderSetting(
                title    = "Speech Pitch",
                value    = ttsPitch,
                range    = 0.5f..2f,
                steps    = 14,
                display  = "%.1f".format(ttsPitch),
                onChange = { vm.updateTtsPitch(it) }
            )
        }

        // ── System Prompt ──────────────────────────────────────────────────────
        SettingsSection("System Prompt") {
            var editedPrompt by remember { mutableStateOf(systemPrompt) }
            LaunchedEffect(systemPrompt) { editedPrompt = systemPrompt }
            OutlinedTextField(
                value    = editedPrompt,
                onValueChange = { editedPrompt = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
                    .heightIn(min = 120.dp),
                label = { Text("System Prompt", color = JarvisTextMuted) },
                shape = RoundedCornerShape(12.dp),
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
            Spacer(Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Button(
                    onClick = { vm.updateSystemPrompt(editedPrompt) },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = JarvisBlue),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(Icons.Default.Save, null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Save")
                }
                OutlinedButton(
                    onClick = {
                        editedPrompt = AppPreferences.DEFAULT_SYSTEM_PROMPT
                        vm.updateSystemPrompt(AppPreferences.DEFAULT_SYSTEM_PROMPT)
                    },
                    modifier = Modifier.height(40.dp),
                    border = BorderStroke(1.dp, JarvisTextMuted.copy(0.5f)),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text("Reset", color = JarvisTextMuted, style = MaterialTheme.typography.labelMedium)
                }
            }
        }

        // ── About ──────────────────────────────────────────────────────────────
        SettingsSection("About") {
            InfoRow("Version", "1.0.0")
            InfoRow("LLM Engine", "llama.cpp")
            InfoRow("STT Engine", "whisper.cpp")
            InfoRow("TTS Engine", "Android TextToSpeech")
        }

        Spacer(Modifier.height(40.dp))
    }
}

@Composable
private fun SettingsSection(title: String, content: @Composable ColumnScope.() -> Unit) {
    Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
        Text(
            title.uppercase(),
            style = MaterialTheme.typography.labelLarge,
            color = JarvisBlue.copy(0.7f),
            modifier = Modifier.padding(bottom = 10.dp, start = 4.dp)
        )
        Surface(
            color = JarvisSurface,
            shape = RoundedCornerShape(16.dp),
            border = BorderStroke(1.dp, JarvisBlueDark.copy(0.2f)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(vertical = 8.dp)) { content() }
        }
    }
}

@Composable
private fun ActionButton(
    icon: ImageVector,
    title: String,
    subtitle: String,
    color: androidx.compose.ui.graphics.Color,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(42.dp)
                .background(color.copy(0.15f), RoundedCornerShape(10.dp)),
            contentAlignment = Alignment.Center
        ) {
            Icon(icon, null, tint = color, modifier = Modifier.size(22.dp))
        }
        Spacer(Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium, color = JarvisText)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = JarvisTextMuted)
        }
        Icon(Icons.Default.OpenInNew, null, tint = JarvisTextMuted, modifier = Modifier.size(18.dp))
    }
}

@Composable
private fun SliderSetting(
    title: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int,
    display: String,
    onChange: (Float) -> Unit
) {
    Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(title, style = MaterialTheme.typography.titleMedium, color = JarvisText)
            Surface(
                color = JarvisBlue.copy(0.15f),
                shape = RoundedCornerShape(8.dp)
            ) {
                Text(display, modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.labelMedium, color = JarvisBlue)
            }
        }
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = range,
            steps = steps,
            modifier = Modifier.fillMaxWidth(),
            colors = SliderDefaults.colors(
                thumbColor       = JarvisBlue,
                activeTrackColor = JarvisBlue,
                inactiveTrackColor = JarvisBlueDark.copy(0.3f)
            )
        )
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = JarvisTextMuted)
        Text(value,  style = MaterialTheme.typography.bodyMedium, color = JarvisText)
    }
}
