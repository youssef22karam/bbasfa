package com.jarvis.app.service

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

class JarvisAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "JarvisA11y"
        var instance: JarvisAccessibilityService? = null
        var lastScreenText: String = ""
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "Accessibility service connected")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return
        // Capture screen text for vision context
        rootInActiveWindow?.let { root ->
            lastScreenText = extractText(root)
            root.recycle()
        }
    }

    override fun onInterrupt() {
        Log.w(TAG, "Accessibility service interrupted")
    }

    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }

    // ── Public actions ────────────────────────────────────────────────────────

    fun executeAction(command: String) {
        val lower = command.lowercase()
        when {
            "back" in lower     -> performGlobalAction(GLOBAL_ACTION_BACK)
            "home" in lower     -> performGlobalAction(GLOBAL_ACTION_HOME)
            "recents" in lower  -> performGlobalAction(GLOBAL_ACTION_RECENTS)
            "notifications" in lower -> performGlobalAction(GLOBAL_ACTION_NOTIFICATIONS)
            "screenshot" in lower    -> performGlobalAction(GLOBAL_ACTION_TAKE_SCREENSHOT)
            "scroll down" in lower   -> performScroll(false)
            "scroll up" in lower     -> performScroll(true)
            "click" in lower    -> findAndClickByText(extractTarget(command, "click"))
            "tap" in lower      -> findAndClickByText(extractTarget(command, "tap"))
            "type" in lower     -> typeText(extractTarget(command, "type"))
            "swipe left" in lower    -> performSwipe(direction = "left")
            "swipe right" in lower   -> performSwipe(direction = "right")
        }
    }

    fun findAndClickByText(text: String): Boolean {
        if (text.isBlank()) return false
        val root = rootInActiveWindow ?: return false
        return try {
            val nodes = root.findAccessibilityNodeInfosByText(text)
            nodes.firstOrNull()?.let { node ->
                val clicked = node.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                if (!clicked) {
                    // Try clicking parent
                    node.parent?.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                }
                node.recycle()
                true
            } ?: false
        } finally {
            root.recycle()
        }
    }

    fun typeText(text: String) {
        if (text.isBlank()) return
        val root = rootInActiveWindow ?: return
        try {
            val focusedNode = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            focusedNode?.let { node ->
                val args = android.os.Bundle()
                args.putString(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
                node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
                node.recycle()
            }
        } finally {
            root.recycle()
        }
    }

    fun clickAt(x: Float, y: Float) {
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 50)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        dispatchGesture(gesture, null, null)
    }

    fun performScroll(up: Boolean) {
        val root = rootInActiveWindow ?: return
        try {
            val scrollable = findScrollableNode(root)
            scrollable?.performAction(
                if (up) AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                else    AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
            )
            scrollable?.recycle()
        } finally {
            root.recycle()
        }
    }

    fun performSwipe(direction: String) {
        val display = resources.displayMetrics
        val w = display.widthPixels.toFloat()
        val h = display.heightPixels.toFloat()
        val path = Path()
        when (direction) {
            "left"  -> { path.moveTo(w * 0.8f, h / 2); path.lineTo(w * 0.2f, h / 2) }
            "right" -> { path.moveTo(w * 0.2f, h / 2); path.lineTo(w * 0.8f, h / 2) }
            "up"    -> { path.moveTo(w / 2, h * 0.8f); path.lineTo(w / 2, h * 0.2f) }
            "down"  -> { path.moveTo(w / 2, h * 0.2f); path.lineTo(w / 2, h * 0.8f) }
            else    -> return
        }
        val stroke = GestureDescription.StrokeDescription(path, 0, 300)
        dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
    }

    fun getScreenContext(): String = lastScreenText

    // ── Private helpers ───────────────────────────────────────────────────────

    private fun extractText(node: AccessibilityNodeInfo): String {
        val sb = StringBuilder()
        extractTextRecursive(node, sb, depth = 0)
        return sb.toString().take(2000)
    }

    private fun extractTextRecursive(node: AccessibilityNodeInfo, sb: StringBuilder, depth: Int) {
        if (depth > 15) return
        node.text?.let { if (it.isNotBlank()) sb.appendLine(it) }
        node.contentDescription?.let { if (it.isNotBlank()) sb.appendLine(it) }
        for (i in 0 until node.childCount) {
            node.getChild(i)?.let { child ->
                extractTextRecursive(child, sb, depth + 1)
                child.recycle()
            }
        }
    }

    private fun findScrollableNode(node: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        if (node.isScrollable) return AccessibilityNodeInfo.obtain(node)
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            val result = findScrollableNode(child)
            child.recycle()
            if (result != null) return result
        }
        return null
    }

    private fun extractTarget(command: String, verb: String): String =
        command.lowercase().substringAfter(verb).trim()
            .removePrefix("on").removePrefix("the").trim()
}
