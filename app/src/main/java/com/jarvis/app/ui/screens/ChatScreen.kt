package com.jarvis.app.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
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
import androidx.compose.ui.draw.*
import androidx.compose.ui.graphics.*
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.*
import com.jarvis.app.data.*
import com.jarvis.app.ui.theme.*
import com.jarvis.app.viewmodel.MainViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: MainViewModel) {
    val messages     by vm.messages.collectAsState()
    val jarvisState  by vm.jarvisState.collectAsState()
    val statusText   by vm.statusText.collectAsState()
    val activeModel  by vm.activeModelName.collectAsState()
    val error        by vm.error.collectAsState()

    var textInput    by remember { mutableStateOf("") }
    val listState    = rememberLazyListState()
    val scope        = rememberCoroutineScope()

    // Auto-scroll
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size - 1)
    }

    // Error snackbar
    val snackbarHost = remember { SnackbarHostState() }
    LaunchedEffect(error) {
        error?.let {
            snackbarHost.showSnackbar(it)
            vm.clearError()
        }
    }

    Scaffold(
        containerColor = JarvisBg,
        snackbarHost   = { SnackbarHost(snackbarHost) }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // ── Header ────────────────────────────────────────────────────────
            JarvisHeader(activeModel = activeModel, statusText = statusText, state = jarvisState)

            // ── Messages ──────────────────────────────────────────────────────
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
                contentPadding = PaddingValues(vertical = 12.dp)
            ) {
                items(messages, key = { it.id }) { msg ->
                    ChatBubble(msg)
                }
            }

            // ── Big Voice Button ──────────────────────────────────────────────
            VoiceControlSection(
                state       = jarvisState,
                onPress     = { vm.startListening() },
                onRelease   = { vm.stopListeningAndProcess() },
                onStop      = { vm.stopGeneration() }
            )

            // ── Text Input ────────────────────────────────────────────────────
            TextInputBar(
                value    = textInput,
                onChange = { textInput = it },
                onSend   = {
                    if (textInput.isNotBlank()) {
                        vm.sendMessage(textInput)
                        textInput = ""
                    }
                },
                onClear  = { vm.clearChat() },
                enabled  = jarvisState == JarvisState.IDLE
            )
        }
    }
}

// ── Header ────────────────────────────────────────────────────────────────────
@Composable
private fun JarvisHeader(activeModel: String, statusText: String, state: JarvisState) {
    val pulse by rememberInfiniteTransition(label = "pulse").animateFloat(
        initialValue = 0.6f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1000), RepeatMode.Reverse),
        label = "alpha"
    )
    val stateColor = when (state) {
        JarvisState.LISTENING    -> JarvisRed
        JarvisState.TRANSCRIBING -> JarvisGold
        JarvisState.THINKING     -> JarvisBlue
        JarvisState.SPEAKING     -> Color(0xFF00FF88)
        JarvisState.ERROR        -> JarvisRed
        JarvisState.IDLE         -> JarvisBlue
    }

    Surface(
        color = JarvisSurface,
        shadowElevation = 4.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 14.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                // Arc Reactor Icon
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .background(JarvisGlowDim, CircleShape)
                        .border(2.dp, stateColor.copy(alpha = if (state != JarvisState.IDLE) pulse else 0.6f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Text("⬡", fontSize = 20.sp, color = stateColor)
                }
                Spacer(Modifier.width(12.dp))
                Column {
                    Text(
                        "J.A.R.V.I.S",
                        style = MaterialTheme.typography.titleLarge,
                        color = JarvisBlue,
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        activeModel,
                        style = MaterialTheme.typography.labelMedium,
                        color = JarvisTextMuted
                    )
                }
                Spacer(Modifier.weight(1f))
                // Status pill
                Surface(
                    color = stateColor.copy(alpha = 0.15f),
                    shape = RoundedCornerShape(20.dp),
                    border = BorderStroke(1.dp, stateColor.copy(alpha = 0.4f))
                ) {
                    Text(
                        statusText,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        style = MaterialTheme.typography.labelMedium,
                        color = stateColor
                    )
                }
            }
        }
    }
}

// ── Chat Bubble ───────────────────────────────────────────────────────────────
@Composable
private fun ChatBubble(msg: ChatMessage) {
    val isUser = msg.role == ChatRole.USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        if (!isUser) {
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .background(JarvisBlue.copy(0.2f), CircleShape)
                    .border(1.dp, JarvisBlue.copy(0.5f), CircleShape)
                    .align(Alignment.Bottom),
                contentAlignment = Alignment.Center
            ) { Text("AI", fontSize = 10.sp, color = JarvisBlue) }
            Spacer(Modifier.width(8.dp))
        }
        Column(
            modifier = Modifier.widthIn(max = 300.dp),
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
        ) {
            Surface(
                color = if (isUser) JarvisBlue.copy(0.2f) else JarvisSurface2,
                shape = RoundedCornerShape(
                    topStart = if (isUser) 18.dp else 4.dp,
                    topEnd   = if (isUser) 4.dp  else 18.dp,
                    bottomStart = 18.dp, bottomEnd = 18.dp
                ),
                border = BorderStroke(
                    1.dp,
                    if (isUser) JarvisBlue.copy(0.4f) else JarvisBlueDark.copy(0.3f)
                )
            ) {
                Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
                    if (msg.isStreaming && msg.content.isBlank()) {
                        ThinkingDots()
                    } else {
                        Text(
                            msg.content,
                            color = JarvisText,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }
        }
        if (isUser) {
            Spacer(Modifier.width(8.dp))
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .background(JarvisGold.copy(0.2f), CircleShape)
                    .border(1.dp, JarvisGold.copy(0.5f), CircleShape)
                    .align(Alignment.Bottom),
                contentAlignment = Alignment.Center
            ) { Text("ME", fontSize = 9.sp, color = JarvisGold) }
        }
    }
}

@Composable
private fun ThinkingDots() {
    val inf = rememberInfiniteTransition(label = "dots")
    val dot1 by inf.animateFloat(0.2f, 1f, infiniteRepeatable(tween(600, delayMillis = 0),   RepeatMode.Reverse), label="d1")
    val dot2 by inf.animateFloat(0.2f, 1f, infiniteRepeatable(tween(600, delayMillis = 200), RepeatMode.Reverse), label="d2")
    val dot3 by inf.animateFloat(0.2f, 1f, infiniteRepeatable(tween(600, delayMillis = 400), RepeatMode.Reverse), label="d3")
    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        listOf(dot1, dot2, dot3).forEach { alpha ->
            Box(Modifier.size(8.dp).background(JarvisBlue.copy(alpha), CircleShape))
        }
    }
}

// ── Voice Control ─────────────────────────────────────────────────────────────
@Composable
private fun VoiceControlSection(
    state: JarvisState,
    onPress: () -> Unit,
    onRelease: () -> Unit,
    onStop: () -> Unit
) {
    val isListening = state == JarvisState.LISTENING
    val isProcessing = state in listOf(JarvisState.THINKING, JarvisState.TRANSCRIBING, JarvisState.SPEAKING)
    val pulse by rememberInfiniteTransition(label = "voice").animateFloat(
        0.8f, 1.2f,
        infiniteRepeatable(tween(700, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "scale"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 20.dp),
        contentAlignment = Alignment.Center
    ) {
        // Outer glow ring for listening
        if (isListening) {
            Box(
                modifier = Modifier
                    .size(130.dp * pulse)
                    .background(JarvisRed.copy(alpha = 0.15f), CircleShape)
            )
            Box(
                modifier = Modifier
                    .size(110.dp)
                    .background(JarvisRed.copy(alpha = 0.1f), CircleShape)
            )
        }

        // Main button
        if (isProcessing) {
            // Stop button
            FilledIconButton(
                onClick = onStop,
                modifier = Modifier.size(90.dp),
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = JarvisRed.copy(0.2f)
                ),
                shape = CircleShape
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Default.Stop, contentDescription = "Stop",
                        tint = JarvisRed, modifier = Modifier.size(36.dp))
                    Text("STOP", fontSize = 10.sp, color = JarvisRed)
                }
            }
        } else {
            // Hold-to-speak button
            val btnColor = if (isListening) JarvisRed else JarvisBlue
            Button(
                onClick = {},
                modifier = Modifier
                    .size(90.dp)
                    .pointerInput(Unit) {
                        awaitPointerEventScope {
                            while (true) {
                                val event = awaitPointerEvent()
                                when (event.type) {
                                    androidx.compose.ui.input.pointer.PointerEventType.Press   -> onPress()
                                    androidx.compose.ui.input.pointer.PointerEventType.Release -> onRelease()
                                    else -> {}
                                }
                            }
                        }
                    },
                shape = CircleShape,
                colors = ButtonDefaults.buttonColors(containerColor = btnColor.copy(0.2f)),
                border = BorderStroke(2.dp, btnColor),
                contentPadding = PaddingValues(0.dp),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp)
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        if (isListening) Icons.Default.MicOff else Icons.Default.Mic,
                        contentDescription = "Mic",
                        tint = btnColor,
                        modifier = Modifier.size(36.dp)
                    )
                    Text(
                        if (isListening) "RELEASE" else "HOLD",
                        fontSize = 10.sp, color = btnColor
                    )
                }
            }
        }
    }
}

// ── Text Input Bar ────────────────────────────────────────────────────────────
@Composable
private fun TextInputBar(
    value: String,
    onChange: (String) -> Unit,
    onSend: () -> Unit,
    onClear: () -> Unit,
    enabled: Boolean
) {
    Surface(
        color = JarvisSurface,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 12.dp, vertical = 10.dp)
                .navigationBarsPadding()
                .imePadding(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Clear
            IconButton(
                onClick = onClear,
                modifier = Modifier.size(44.dp)
            ) {
                Icon(Icons.Default.DeleteOutline, "Clear", tint = JarvisTextMuted)
            }
            // Text field
            OutlinedTextField(
                value = value,
                onValueChange = onChange,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Type a message…", color = JarvisTextMuted) },
                enabled = enabled,
                maxLines = 3,
                shape = RoundedCornerShape(24.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor   = JarvisBlue,
                    unfocusedBorderColor = JarvisBlueDark.copy(0.5f),
                    focusedTextColor     = JarvisText,
                    unfocusedTextColor   = JarvisText,
                    cursorColor          = JarvisBlue,
                    focusedContainerColor   = JarvisSurface2,
                    unfocusedContainerColor = JarvisSurface2
                )
            )
            // Send
            FilledIconButton(
                onClick = onSend,
                enabled = enabled && value.isNotBlank(),
                modifier = Modifier.size(48.dp),
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = JarvisBlue,
                    disabledContainerColor = JarvisBlueDark.copy(0.3f)
                )
            ) {
                Icon(Icons.Default.Send, "Send", tint = Color(0xFF001F29))
            }
        }
    }
}

// Required import for pointerInput
private fun Modifier.pointerInput(
    key: Any,
    block: suspend androidx.compose.ui.input.pointer.PointerInputScope.() -> Unit
): Modifier = this.then(
    androidx.compose.ui.input.pointer.pointerInput(key, block)
)
