package com.jarvis.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.compose.material3.Typography

val JarvisTypography = Typography(
    displayLarge  = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Bold,   fontSize = 32.sp, color = JarvisBlue),
    headlineLarge = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.SemiBold, fontSize = 24.sp),
    headlineMedium= TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.SemiBold, fontSize = 20.sp),
    titleLarge    = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.SemiBold, fontSize = 18.sp),
    titleMedium   = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Medium,  fontSize = 16.sp),
    bodyLarge     = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Normal,  fontSize = 16.sp),
    bodyMedium    = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Normal,  fontSize = 14.sp),
    labelLarge    = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Medium,  fontSize = 14.sp),
    labelMedium   = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Medium,  fontSize = 12.sp),
)

@Composable
fun JarvisTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = JarvisDarkColorScheme,
        typography  = JarvisTypography,
        content     = content
    )
}
