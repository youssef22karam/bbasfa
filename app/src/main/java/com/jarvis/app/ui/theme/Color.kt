package com.jarvis.app.ui.theme

import androidx.compose.material3.*
import androidx.compose.ui.graphics.Color

// ── JARVIS Color Palette ──────────────────────────────────────────────────────
val JarvisBlue        = Color(0xFF00D4FF)   // Arc reactor cyan
val JarvisBlueLight   = Color(0xFF80EAFF)
val JarvisBlueDark    = Color(0xFF007A99)
val JarvisGold        = Color(0xFFFFB300)   // Iron Man gold
val JarvisRed         = Color(0xFFEF233C)   // Iron Man red
val JarvisBg          = Color(0xFF050D1A)   // Deep space background
val JarvisSurface     = Color(0xFF0A1628)   // Panel surface
val JarvisSurface2    = Color(0xFF0F2040)   // Elevated surface
val JarvisGlowDim     = Color(0x3300D4FF)   // Glow tint
val JarvisText        = Color(0xFFE8F4FF)
val JarvisTextMuted   = Color(0xFF6B8FA8)

val JarvisDarkColorScheme = darkColorScheme(
    primary          = JarvisBlue,
    onPrimary        = Color(0xFF001F29),
    primaryContainer = JarvisBlueDark,
    onPrimaryContainer = JarvisBlueLight,
    secondary        = JarvisGold,
    onSecondary      = Color(0xFF1A0F00),
    tertiary         = JarvisRed,
    background       = JarvisBg,
    onBackground     = JarvisText,
    surface          = JarvisSurface,
    onSurface        = JarvisText,
    surfaceVariant   = JarvisSurface2,
    onSurfaceVariant = JarvisTextMuted,
    error            = JarvisRed,
    outline          = JarvisBlueDark
)
