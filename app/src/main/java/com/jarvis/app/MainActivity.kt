package com.jarvis.app

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.*
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.*
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.*
import com.jarvis.app.navigation.Screen
import com.jarvis.app.service.JarvisService
import com.jarvis.app.ui.screens.*
import com.jarvis.app.ui.theme.*
import com.jarvis.app.viewmodel.MainViewModel

class MainActivity : ComponentActivity() {

    private val vm: MainViewModel by viewModels()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* results handled silently */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestPermissions()
        startJarvisService()

        setContent {
            JarvisTheme {
                JarvisApp(vm)
            }
        }
    }

    private fun requestPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.RECORD_AUDIO
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions += Manifest.permission.POST_NOTIFICATIONS
            permissions += Manifest.permission.READ_MEDIA_AUDIO
            permissions += Manifest.permission.READ_MEDIA_IMAGES
        } else {
            permissions += Manifest.permission.READ_EXTERNAL_STORAGE
            permissions += Manifest.permission.WRITE_EXTERNAL_STORAGE
        }
        permissionLauncher.launch(permissions.toTypedArray())
    }

    private fun startJarvisService() {
        val intent = Intent(this, JarvisService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun JarvisApp(vm: MainViewModel) {
    val navController = rememberNavController()

    val navItems = listOf(
        Triple(Screen.Chat,     Icons.Default.Chat,         "Chat"),
        Triple(Screen.Models,   Icons.Default.Storage,      "Models"),
        Triple(Screen.Download, Icons.Default.CloudDownload,"Download"),
        Triple(Screen.Settings, Icons.Default.Settings,     "Settings")
    )

    Scaffold(
        containerColor = JarvisBg,
        bottomBar = {
            NavigationBar(
                containerColor = JarvisSurface,
                tonalElevation = 0.dp
            ) {
                val navBackStack by navController.currentBackStackEntryAsState()
                val current = navBackStack?.destination
                navItems.forEach { (screen, icon, label) ->
                    val selected = current?.hierarchy?.any { it.route == screen.route } == true
                    NavigationBarItem(
                        selected = selected,
                        onClick  = {
                            navController.navigate(screen.route) {
                                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon  = {
                            Icon(
                                icon, label,
                                modifier = Modifier.size(if (selected) 26.dp else 22.dp)
                            )
                        },
                        label = { Text(label, style = MaterialTheme.typography.labelSmall) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor   = JarvisBlue,
                            selectedTextColor   = JarvisBlue,
                            unselectedIconColor = JarvisTextMuted,
                            unselectedTextColor = JarvisTextMuted,
                            indicatorColor      = JarvisBlue.copy(0.15f)
                        )
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Chat.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Chat.route)     { ChatScreen(vm) }
            composable(Screen.Models.route)   { ModelsScreen(vm) }
            composable(Screen.Download.route) { DownloadScreen(vm) }
            composable(Screen.Settings.route) { SettingsScreen(vm) }
        }
    }
}
