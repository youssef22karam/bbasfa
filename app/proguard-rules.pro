-keep class com.jarvis.app.engine.** { *; }
-keep class com.jarvis.app.data.** { *; }
-keepclassmembers class * {
    native <methods>;
}
