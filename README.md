# ⬡ JARVIS AI — Local LLM Assistant for Android

> Run powerful AI models **entirely on your phone**. Voice input via Whisper, voice output via TTS, full phone control via Accessibility Services. No cloud, no API keys.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 Local LLM | Run any GGUF model (Qwen, Llama, Phi, Gemma…) via **llama.cpp** |
| 🎙️ Voice Input | **Whisper.cpp** transcribes your speech to text on-device |
| 🔊 Voice Output | Android TTS reads every response aloud |
| 📱 Phone Control | Accessibility Service taps buttons, types text, scrolls, swipes |
| 🌐 HuggingFace Browser | Search & download GGUF models directly inside the app |
| 📥 Import / Export | Move models in/out via file manager — no re-downloading |
| 🔋 Background | Foreground service keeps JARVIS alive while you use other apps |
| 🌑 JARVIS Dark Theme | Arc-reactor cyan + Iron Man gold design |

---

## 📲 Getting the APK (no coding needed)

### Step 1 — Fork this repository
1. Open **https://github.com** and sign in (create a free account if needed)
2. Click **Fork** (top-right) on this repo — this copies it to your account

### Step 2 — Trigger the build
1. In **your fork**, click the **Actions** tab
2. If you see "Workflows aren't running", click **"I understand my workflows, enable them"**
3. In the left sidebar click **Build JarvisAI APK**
4. Click **Run workflow → Run workflow** (green button)
5. Wait **20–40 minutes** for the build to finish ☕

### Step 3 — Download the APK
1. When the build turns ✅ green, click on the run
2. Scroll to the bottom → **Artifacts** section
3. Click **JarvisAI-debug-apk** to download a `.zip`
4. Unzip it on your phone → you get `app-debug.apk`

### Step 4 — Install on Android
1. On your phone: **Settings → Security → Install unknown apps** → allow your file manager
2. Tap `app-debug.apk` → **Install**

---

## 🚀 First-Time Setup (inside the app)

### 1. Grant permissions
The app will ask for:
- **Microphone** — for voice input (required)
- **Notifications** — for background service
- **Files** — for model import/export

### 2. Enable Accessibility Service (for phone control)
1. Go to **Settings tab** in the app
2. Tap **"Enable Accessibility Service"**
3. Find **JARVIS Phone Control** in the list
4. Toggle it **ON**
5. Tap **Allow**

### 3. Download a model
Go to the **Download tab**:

| Model | Size | Use case |
|---|---|---|
| Qwen 2.5 0.5B Q4_K_M | ~400 MB | Fastest, basic tasks |
| Qwen 2.5 1.5B Q4_K_M | ~1 GB | Good balance |
| Llama 3.2 1B Q4_K_M | ~700 MB | Strong reasoning |
| Llama 3.2 3B Q4_K_M | ~2 GB | Best quality |
| Phi-3.5 Mini Q4_K_M | ~2.2 GB | Excellent for phone use |

For **Whisper STT**, search `ggerganov/whisper.cpp` and download `ggml-small.bin` (~500 MB) or `ggml-tiny.bin` (~75 MB).

### 4. Load the model
- Go to **Models tab** → tap **Load** on your downloaded model
- Tap **Set STT** on your Whisper model

### 5. Talk to JARVIS!
- **Hold** the blue mic button to speak
- **Release** to transcribe and get a response
- Or type in the text bar

---

## 📁 Import/Export Models

### Import (from your phone storage or PC)
1. Go to **Models tab**
2. Tap **Import LLM** (for language models) or **Import Whisper**
3. Navigate to your `.gguf` or `.bin` file
4. Done — it copies into the app's private storage

### Export (to share or backup)
1. Go to **Models tab**
2. Find your model → tap the **⬇ export button**
3. Choose where to save (Files, Google Drive, etc.)

### Transfer from PC
Connect your phone via USB, then copy `.gguf` files to your phone storage. Then use **Import** in the app.

---

## 🎙️ Voice Commands for Phone Control

After enabling Accessibility Service, JARVIS understands:

```
"Go back"              → presses Back
"Go home"              → presses Home
"Show recents"         → opens recent apps
"Take a screenshot"    → takes screenshot
"Scroll down"          → scrolls the screen
"Click [button name]"  → taps a button by label
"Type [text]"          → types in the focused field
"Swipe left/right"     → swipes the screen
```

---

## ⚙️ Settings Reference

| Setting | Default | Description |
|---|---|---|
| Max Tokens | 512 | Maximum response length |
| Temperature | 0.7 | Creativity (0=focused, 2=creative) |
| CPU Threads | 4 | More = faster (but uses more battery) |
| Speech Rate | 1.0x | How fast JARVIS speaks |
| Speech Pitch | 0.95 | Voice pitch |
| System Prompt | JARVIS persona | What JARVIS thinks it is |

---

## 📱 Recommended Phones

| RAM | Recommended Models |
|---|---|
| 4 GB | Qwen 0.5B, Qwen 1.5B |
| 6 GB | Llama 3.2 1B, Qwen 1.5B |
| 8 GB | Llama 3.2 3B, Phi-3.5 Mini |
| 12 GB+ | Any model up to ~7B |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│              Android App                │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐  │
│  │  Chat   │ │ Download │ │Settings │  │
│  │ Screen  │ │ Screen   │ │ Screen  │  │
│  └────┬────┘ └─────┬────┘ └────┬────┘  │
│       └────────────┼───────────┘        │
│              MainViewModel              │
│       ┌───────────┬┴──────────┐         │
│  LlamaEngine  WhisperEngine  TTS        │
│       │           │                     │
│  ┌────┴────┐ ┌────┴────┐                │
│  │llama.cpp│ │whisper  │   JNI (C++)    │
│  │  (JNI) │ │  .cpp   │                │
│  └─────────┘ └─────────┘                │
│                                         │
│  JarvisAccessibilityService             │
│  (taps, scrolls, types, reads screen)   │
│                                         │
│  JarvisService (foreground, background) │
└─────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

**"No model loaded" error**
→ Go to Models tab and tap Load on a model

**Voice button does nothing**
→ Make sure you granted Microphone permission (Settings → Apps → JARVIS → Permissions)

**Whisper not transcribing**
→ Download and set a Whisper model in the Models tab

**Phone control not working**
→ Go to Settings → Enable Accessibility Service → turn on JARVIS Phone Control

**Build fails on GitHub Actions**
→ Check the Actions log for errors. Common fix: make sure you enabled Actions in the fork.

**App crashes on launch**
→ Make sure your phone runs Android 8.0+ (API 26+)

**Model loads slowly**
→ Increase CPU Threads in Settings (try 6-8 on modern phones)

---

## 📄 License

MIT License — use freely, modify, redistribute.

Built with [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — both MIT licensed.
