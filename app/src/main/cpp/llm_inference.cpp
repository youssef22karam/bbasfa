#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <sstream>

#include "llama.h"

#define TAG "JarvisLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ─────────────────────────────────────────────────────────────────────────────
// Handle struct
// ─────────────────────────────────────────────────────────────────────────────
struct LlamaHandle {
    llama_model   *model   = nullptr;
    llama_context *ctx     = nullptr;
    std::atomic<bool> stop_flag{false};
};

static std::string jstring2str(JNIEnv *env, jstring jstr) {
    if (!jstr) return {};
    const char *chars = env->GetStringUTFChars(jstr, nullptr);
    std::string s(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return s;
}

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// nativeLoadModel(path, nThreads, nCtx) → handle (0 on failure)
// ─────────────────────────────────────────────────────────────────────────────
JNIEXPORT jlong JNICALL
Java_com_jarvis_app_engine_LlamaEngine_nativeLoadModel(
        JNIEnv *env, jobject /*thiz*/,
        jstring modelPath, jint nThreads, jint nCtx) {

    std::string path = jstring2str(env, modelPath);
    LOGI("Loading model: %s  threads=%d  ctx=%d", path.c_str(), nThreads, nCtx);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only on most Android devices

    llama_model *model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) {
        LOGE("Failed to load model from: %s", path.c_str());
        return 0L;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = static_cast<uint32_t>(nCtx);
    cparams.n_threads = static_cast<uint32_t>(nThreads);
    cparams.n_threads_batch = static_cast<uint32_t>(nThreads);

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOGE("Failed to create context");
        llama_model_free(model);
        return 0L;
    }

    auto *handle = new LlamaHandle{model, ctx};
    LOGI("Model loaded OK, handle=%p", handle);
    return reinterpret_cast<jlong>(handle);
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeGenerate — streams tokens via callback object
// ─────────────────────────────────────────────────────────────────────────────
JNIEXPORT jint JNICALL
Java_com_jarvis_app_engine_LlamaEngine_nativeGenerate(
        JNIEnv *env, jobject /*thiz*/,
        jlong handlePtr, jstring jPrompt,
        jint maxTokens, jfloat temperature,
        jobject callback) {

    auto *h = reinterpret_cast<LlamaHandle *>(handlePtr);
    if (!h || !h->ctx) return -1;

    h->stop_flag.store(false);

    std::string prompt = jstring2str(env, jPrompt);

    // Tokenise
    const bool add_bos = llama_add_bos_token(h->model);
    std::vector<llama_token> tokens_list = llama_tokenize(
            h->ctx, prompt, add_bos, true);

    if (tokens_list.empty()) { return -2; }

    // KV-cache reset
    llama_kv_cache_clear(h->ctx);

    // Prefill batch
    llama_batch batch = llama_batch_get_one(tokens_list.data(),
                                             (int)tokens_list.size());
    if (llama_decode(h->ctx, batch) != 0) {
        LOGE("llama_decode (prefill) failed");
        return -3;
    }

    // Sampler chain
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Prepare callback method lookup
    jclass  cbClass    = env->GetObjectClass(callback);
    jmethodID onToken  = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)Z");

    int n_generated = 0;
    while (n_generated < maxTokens) {
        if (h->stop_flag.load()) break;

        llama_token id = llama_sampler_sample(smpl, h->ctx, -1);
        if (llama_token_is_eog(h->model, id)) break;

        // Detokenise
        char buf[256];
        int  len = llama_token_to_piece(h->model, id, buf, sizeof(buf), 0, true);
        if (len < 0) break;
        buf[len] = '\0';

        // Fire callback → Kotlin
        jstring piece = env->NewStringUTF(buf);
        jboolean cont = env->CallBooleanMethod(callback, onToken, piece);
        env->DeleteLocalRef(piece);
        if (!cont || env->ExceptionCheck()) break;

        // Next-token batch
        llama_batch nb = llama_batch_get_one(&id, 1);
        if (llama_decode(h->ctx, nb) != 0) break;

        ++n_generated;
    }

    llama_sampler_free(smpl);
    return n_generated;
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeStopGeneration
// ─────────────────────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_jarvis_app_engine_LlamaEngine_nativeStopGeneration(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong handlePtr) {
    if (auto *h = reinterpret_cast<LlamaHandle *>(handlePtr))
        h->stop_flag.store(true);
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeFreeModel
// ─────────────────────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_jarvis_app_engine_LlamaEngine_nativeFreeModel(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong handlePtr) {
    auto *h = reinterpret_cast<LlamaHandle *>(handlePtr);
    if (!h) return;
    h->stop_flag.store(true);
    if (h->ctx)   llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
    LOGI("Model freed");
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeGetModelDescription
// ─────────────────────────────────────────────────────────────────────────────
JNIEXPORT jstring JNICALL
Java_com_jarvis_app_engine_LlamaEngine_nativeGetModelDescription(
        JNIEnv *env, jobject /*thiz*/, jlong handlePtr) {
    auto *h = reinterpret_cast<LlamaHandle *>(handlePtr);
    if (!h || !h->model) return env->NewStringUTF("(no model)");

    char buf[512];
    llama_model_desc(h->model, buf, sizeof(buf));
    return env->NewStringUTF(buf);
}

} // extern "C"
