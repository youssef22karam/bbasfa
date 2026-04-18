#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <cstring>

#include "whisper.h"

#define TAG "JarvisWhisper"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// nativeInitContext(modelPath) → context pointer
// ─────────────────────────────────────────────────────────────────────────────
JNIEXPORT jlong JNICALL
Java_com_jarvis_app_engine_WhisperEngine_nativeInitContext(
        JNIEnv *env, jobject /*thiz*/, jstring jModelPath) {

    const char *path = env->GetStringUTFChars(jModelPath, nullptr);
    LOGI("Loading Whisper model: %s", path);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    struct whisper_context *ctx = whisper_init_from_file_with_params(path, cparams);
    env->ReleaseStringUTFChars(jModelPath, path);

    if (!ctx) {
        LOGE("Failed to load Whisper model");
        return 0L;
    }
    LOGI("Whisper context created OK");
    return reinterpret_cast<jlong>(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeFreeContext
// ─────────────────────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_jarvis_app_engine_WhisperEngine_nativeFreeContext(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong ctxPtr) {
    if (auto *ctx = reinterpret_cast<whisper_context *>(ctxPtr))
        whisper_free(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeTranscribe(ctxPtr, pcmFloat32Array, language) → transcript string
// ─────────────────────────────────────────────────────────────────────────────
JNIEXPORT jstring JNICALL
Java_com_jarvis_app_engine_WhisperEngine_nativeTranscribe(
        JNIEnv *env, jobject /*thiz*/,
        jlong ctxPtr, jfloatArray jAudio, jstring jLang) {

    auto *ctx = reinterpret_cast<whisper_context *>(ctxPtr);
    if (!ctx) return env->NewStringUTF("[error: no context]");

    jsize n_samples = env->GetArrayLength(jAudio);
    float *samples  = env->GetFloatArrayElements(jAudio, nullptr);

    const char *lang = env->GetStringUTFChars(jLang, nullptr);

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime   = false;
    params.print_progress   = false;
    params.print_timestamps = false;
    params.language         = lang;
    params.n_threads        = 4;
    params.single_segment   = false;
    params.no_context       = true;

    int rc = whisper_full(ctx, params, samples, (int)n_samples);

    env->ReleaseFloatArrayElements(jAudio, samples, JNI_ABORT);
    env->ReleaseStringUTFChars(jLang, lang);

    if (rc != 0) {
        LOGE("whisper_full failed: %d", rc);
        return env->NewStringUTF("");
    }

    // Concatenate all segments
    std::string result;
    int n_segs = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segs; i++) {
        const char *txt = whisper_full_get_segment_text(ctx, i);
        if (txt) result += txt;
    }

    // Trim leading/trailing whitespace
    size_t start = result.find_first_not_of(" \t\n\r");
    size_t end   = result.find_last_not_of(" \t\n\r");
    if (start != std::string::npos)
        result = result.substr(start, end - start + 1);

    return env->NewStringUTF(result.c_str());
}

} // extern "C"
