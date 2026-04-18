package com.jarvis.app.navigation

sealed class Screen(val route: String) {
    object Chat     : Screen("chat")
    object Models   : Screen("models")
    object Download : Screen("download")
    object Settings : Screen("settings")
}
