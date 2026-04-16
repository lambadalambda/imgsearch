#include "bridge.h"

#ifdef IMGSEARCH_LLAMA_NATIVE_ENABLED

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include "../../../deps/llama.cpp/common/json-schema-to-grammar.h"
#include "../../../deps/llama.cpp/ggml/include/ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../../../deps/llama.cpp/vendor/nlohmann/json.hpp"

#ifdef __APPLE__
#include <mach-o/dyld.h>
#elif !defined(_WIN32)
#include <unistd.h>
#endif

struct imgsearch_llama_handle {
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    mtmd_context * mctx = nullptr;
    int32_t dims = 0;
    int32_t n_batch = 512;
    int32_t image_max_side = 512;
    std::string last_error;
    common_chat_templates_ptr tmpls;
};

namespace {

namespace fs = std::filesystem;

constexpr int32_t kDefaultImageMaxSide = 512;

std::once_flag g_backend_once;
std::mutex g_error_mu;
std::string g_last_error;

bool should_drop_log(ggml_log_level level, const char * text) {
    if (text == nullptr) {
        return true;
    }
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return true;
    }

    return std::strstr(text, "embeddings required but some input tokens were not marked as outputs -> overriding") != nullptr ||
           std::strstr(text, "add_text:") != nullptr ||
           std::strstr(text, "image_tokens->nx =") != nullptr ||
           std::strstr(text, "image_tokens->ny =") != nullptr ||
           std::strstr(text, "batch_f32 size =") != nullptr;
}

void native_log_callback(ggml_log_level level, const char * text, void *) {
    if (should_drop_log(level, text)) {
        return;
    }
    fputs(text, stderr);
}

std::string executable_dir() {
#ifdef _WIN32
    return "";
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    if (size == 0) {
        return "";
    }
    std::vector<char> path(size + 1, '\0');
    if (_NSGetExecutablePath(path.data(), &size) != 0) {
        return "";
    }
    std::error_code ec;
    fs::path resolved = fs::weakly_canonical(fs::path(path.data()), ec);
    if (ec) {
        resolved = fs::path(path.data());
    }
    return resolved.parent_path().string();
#else
    std::vector<char> path(4096, '\0');
    const auto len = readlink("/proc/self/exe", path.data(), path.size() - 1);
    if (len <= 0) {
        return "";
    }
    path[static_cast<size_t>(len)] = '\0';
    return fs::path(path.data()).parent_path().string();
#endif
}

void load_backends() {
    ggml_backend_load_all();

    if (ggml_backend_reg_count() > 0) {
        return;
    }

    const auto exe_dir = executable_dir();
    if (exe_dir.empty()) {
        return;
    }

    ggml_backend_load_all_from_path(exe_dir.c_str());
    if (ggml_backend_reg_count() > 0) {
        return;
    }

    const auto lib_dir = (fs::path(exe_dir) / "lib").string();
    ggml_backend_load_all_from_path(lib_dir.c_str());
}

void set_global_error(const std::string & msg) {
    std::lock_guard<std::mutex> lock(g_error_mu);
    g_last_error = msg;
}

int32_t set_error(imgsearch_llama_handle * handle, const std::string & msg) {
    if (handle != nullptr) {
        handle->last_error = msg;
    }
    set_global_error(msg);
    return -1;
}

std::string trim(const char * in) {
    if (in == nullptr) {
        return "";
    }
    std::string s(in);
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string compose_prompt(const char * instruction, const char * content) {
    const std::string inst = trim(instruction);
    const std::string body = trim(content);

    const std::string system_message = inst.empty() ? "Represent the user's input." : inst;

    std::string prompt;
    prompt.reserve(system_message.size() + body.size() + 96);
    prompt += "<|im_start|>system\n";
    prompt += system_message;
    prompt += "<|im_end|>\n<|im_start|>user\n";
    prompt += body;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

void normalize_l2(float * out, int32_t n) {
    if (out == nullptr || n <= 0) {
        return;
    }

    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) {
        sum += static_cast<double>(out[i]) * static_cast<double>(out[i]);
    }

    if (sum <= 0.0) {
        return;
    }

    const float inv = static_cast<float>(1.0 / std::sqrt(sum));
    for (int32_t i = 0; i < n; i++) {
        out[i] *= inv;
    }
}

bool resize_rgb_bilinear(
    const unsigned char * src,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h,
    std::vector<unsigned char> & dst) {
    if (src == nullptr || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return false;
    }

    dst.assign(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3, 0);

    for (int32_t y = 0; y < dst_h; y++) {
        const float src_y = (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(dst_h) - 0.5f;
        int32_t y0 = static_cast<int32_t>(std::floor(src_y));
        y0 = std::max<int32_t>(0, std::min<int32_t>(y0, src_h - 1));
        const int32_t y1 = std::min<int32_t>(y0 + 1, src_h - 1);
        const float wy = std::min<float>(1.0f, std::max<float>(0.0f, src_y - static_cast<float>(y0)));

        for (int32_t x = 0; x < dst_w; x++) {
            const float src_x = (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(dst_w) - 0.5f;
            int32_t x0 = static_cast<int32_t>(std::floor(src_x));
            x0 = std::max<int32_t>(0, std::min<int32_t>(x0, src_w - 1));
            const int32_t x1 = std::min<int32_t>(x0 + 1, src_w - 1);
            const float wx = std::min<float>(1.0f, std::max<float>(0.0f, src_x - static_cast<float>(x0)));

            for (int32_t c = 0; c < 3; c++) {
                const float v00 = src[(static_cast<size_t>(y0) * src_w + x0) * 3 + c];
                const float v01 = src[(static_cast<size_t>(y0) * src_w + x1) * 3 + c];
                const float v10 = src[(static_cast<size_t>(y1) * src_w + x0) * 3 + c];
                const float v11 = src[(static_cast<size_t>(y1) * src_w + x1) * 3 + c];
                const float v0 = v00 + (v01 - v00) * wx;
                const float v1 = v10 + (v11 - v10) * wx;
                const float val = v0 + (v1 - v0) * wy;

                dst[(static_cast<size_t>(y) * dst_w + x) * 3 + c] = static_cast<unsigned char>(std::round(std::min<float>(255.0f, std::max<float>(0.0f, val))));
            }
        }
    }

    return true;
}

mtmd_bitmap * maybe_resize_bitmap(imgsearch_llama_handle * handle, mtmd_bitmap * bitmap) {
    if (handle == nullptr || bitmap == nullptr || handle->image_max_side <= 0) {
        return bitmap;
    }

    const uint32_t src_w_u32 = mtmd_bitmap_get_nx(bitmap);
    const uint32_t src_h_u32 = mtmd_bitmap_get_ny(bitmap);
    if (src_w_u32 == 0 || src_h_u32 == 0) {
        mtmd_bitmap_free(bitmap);
        set_error(handle, "invalid decoded image dimensions");
        return nullptr;
    }

    const int32_t src_w = static_cast<int32_t>(src_w_u32);
    const int32_t src_h = static_cast<int32_t>(src_h_u32);
    const int32_t max_side = handle->image_max_side;
    if (src_w <= max_side && src_h <= max_side) {
        return bitmap;
    }

    const float scale = std::min(
        static_cast<float>(max_side) / static_cast<float>(src_w),
        static_cast<float>(max_side) / static_cast<float>(src_h));
    const int32_t dst_w = std::max<int32_t>(1, static_cast<int32_t>(std::floor(src_w * scale)));
    const int32_t dst_h = std::max<int32_t>(1, static_cast<int32_t>(std::floor(src_h * scale)));

    const unsigned char * src = mtmd_bitmap_get_data(bitmap);
    if (src == nullptr) {
        mtmd_bitmap_free(bitmap);
        set_error(handle, "decoded image buffer is empty");
        return nullptr;
    }

    std::vector<unsigned char> resized;
    if (!resize_rgb_bilinear(src, src_w, src_h, dst_w, dst_h, resized)) {
        mtmd_bitmap_free(bitmap);
        set_error(handle, "failed to resize image for llama-cpp-native embedding");
        return nullptr;
    }

    mtmd_bitmap * resized_bitmap = mtmd_bitmap_init(
        static_cast<uint32_t>(dst_w),
        static_cast<uint32_t>(dst_h),
        resized.data());
    mtmd_bitmap_free(bitmap);
    if (resized_bitmap == nullptr) {
        set_error(handle, "failed to allocate resized image bitmap for llama-cpp-native embedding");
        return nullptr;
    }

    return resized_bitmap;
}

int32_t copy_embedding(imgsearch_llama_handle * handle, llama_seq_id seq_id, float * out, int32_t out_len) {
    if (handle == nullptr || handle->lctx == nullptr) {
        return set_error(handle, "llama.cpp handle is not initialized");
    }
    if (out == nullptr || out_len <= 0) {
        return set_error(handle, "embedding output buffer is invalid");
    }
    if (out_len != handle->dims) {
        return set_error(handle, "embedding output buffer dimension mismatch");
    }

    const float * embd = llama_get_embeddings_seq(handle->lctx, seq_id);
    if (embd == nullptr) {
        if (seq_id == 0) {
            embd = llama_get_embeddings_ith(handle->lctx, -1);
        }
    }
    if (embd == nullptr) {
        return set_error(handle, "failed to read llama.cpp embedding output");
    }

    std::memcpy(out, embd, static_cast<size_t>(handle->dims) * sizeof(float));
    normalize_l2(out, handle->dims);
    return 0;
}

int32_t eval_prompt(
    imgsearch_llama_handle * handle,
    const std::string & prompt,
    const mtmd_bitmap ** bitmaps,
    size_t n_bitmaps,
    llama_seq_id seq_id,
    float * out,
    int32_t out_len) {
    if (handle == nullptr || handle->mctx == nullptr || handle->lctx == nullptr) {
        return set_error(handle, "llama.cpp handle is not initialized");
    }

    llama_set_embeddings(handle->lctx, true);
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        return set_error(handle, "failed to allocate mtmd chunks");
    }

    mtmd_input_text text{};
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    int32_t tok_res = mtmd_tokenize(handle->mctx, chunks, &text, bitmaps, n_bitmaps);
    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        return set_error(handle, "mtmd tokenization failed");
    }

    llama_memory_clear(llama_get_memory(handle->lctx), true);

    llama_pos new_n_past = 0;
    int32_t eval_res = mtmd_helper_eval_chunks(
        handle->mctx,
        handle->lctx,
        chunks,
        0,
        seq_id,
        handle->n_batch,
        true,
        &new_n_past);

    mtmd_input_chunks_free(chunks);

    if (eval_res != 0) {
        return set_error(handle, "mtmd evaluation failed");
    }

    return copy_embedding(handle, seq_id, out, out_len);
}

int32_t inspect_prompt(
    imgsearch_llama_handle * handle,
    const std::string & prompt,
    const mtmd_bitmap ** bitmaps,
    size_t n_bitmaps,
    imgsearch_llama_embed_inspect * out) {
    if (handle == nullptr || handle->mctx == nullptr || handle->lctx == nullptr) {
        return set_error(handle, "llama.cpp handle is not initialized");
    }
    if (out == nullptr) {
        return set_error(handle, "inspect output is required");
    }

    std::memset(out, 0, sizeof(*out));

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        return set_error(handle, "failed to allocate mtmd chunks");
    }

    mtmd_input_text text{};
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    int32_t tok_res = mtmd_tokenize(handle->mctx, chunks, &text, bitmaps, n_bitmaps);
    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        return set_error(handle, "mtmd tokenization failed");
    }

    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (chunk == nullptr) {
            continue;
        }

        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        switch (mtmd_input_chunk_get_type(chunk)) {
            case MTMD_INPUT_CHUNK_TYPE_TEXT:
                out->text_chunks++;
                out->text_tokens += static_cast<int32_t>(n_tokens);
                break;
            case MTMD_INPUT_CHUNK_TYPE_IMAGE: {
                out->image_chunks++;
                out->image_tokens += static_cast<int32_t>(n_tokens);
                if (static_cast<int32_t>(n_tokens) > out->max_image_tokens) {
                    out->max_image_tokens = static_cast<int32_t>(n_tokens);
                }

                const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
                if (image_tokens != nullptr) {
                    const int32_t nx = static_cast<int32_t>(mtmd_image_tokens_get_nx(image_tokens));
                    const int32_t ny = static_cast<int32_t>(mtmd_image_tokens_get_ny(image_tokens));
                    if (nx > out->max_image_nx) {
                        out->max_image_nx = nx;
                    }
                    if (ny > out->max_image_ny) {
                        out->max_image_ny = ny;
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    out->total_tokens = static_cast<int32_t>(mtmd_helper_get_n_tokens(chunks));
    out->total_positions = static_cast<int32_t>(mtmd_helper_get_n_pos(chunks));
    out->n_ctx = static_cast<int32_t>(llama_n_ctx(handle->lctx));
    out->n_ctx_seq = static_cast<int32_t>(llama_n_ctx_seq(handle->lctx));
    out->n_seq_max = static_cast<int32_t>(llama_n_seq_max(handle->lctx));
    out->n_batch = static_cast<int32_t>(llama_n_batch(handle->lctx));

    mtmd_input_chunks_free(chunks);
    return 0;
}

bool ensure_chat_templates(imgsearch_llama_handle * handle) {
    if (handle == nullptr || handle->model == nullptr) {
        set_error(handle, "llama.cpp handle is not initialized");
        return false;
    }
    if (handle->tmpls) {
        return true;
    }
    if (!llama_model_chat_template(handle->model, nullptr)) {
        set_error(handle, "model does not provide a chat template required for generation");
        return false;
    }

    try {
        handle->tmpls = common_chat_templates_init(handle->model, "");
    } catch (const std::exception & e) {
        set_error(handle, std::string("failed to initialize chat template: ") + e.what());
        return false;
    }

    if (!handle->tmpls) {
        set_error(handle, "failed to initialize chat template");
        return false;
    }

    return true;
}

int32_t generate_image(
    imgsearch_llama_handle * handle,
    const std::string & image,
    const std::string & system_prompt,
    const std::string & user_prompt,
    const std::string & json_schema,
    int32_t max_tokens,
    float temperature,
    float top_p,
    char * out,
    int32_t out_len) {
    if (handle == nullptr || handle->mctx == nullptr || handle->lctx == nullptr) {
        return set_error(handle, "llama.cpp handle is not initialized");
    }
    if (image.empty()) {
        return set_error(handle, "image_path is required");
    }
    if (out == nullptr || out_len <= 1) {
        return set_error(handle, "generation output buffer is invalid");
    }
    if (max_tokens <= 0) {
        return set_error(handle, "max_tokens must be positive");
    }
    if (!ensure_chat_templates(handle)) {
        return -1;
    }

    mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_file(handle->mctx, image.c_str());
    if (bitmap == nullptr) {
        return set_error(handle, "failed to decode image for mtmd input");
    }

    bitmap = maybe_resize_bitmap(handle, bitmap);
    if (bitmap == nullptr) {
        return -1;
    }

    common_chat_msg user_msg;
    user_msg.role = "user";
    if (!user_prompt.empty()) {
        user_msg.content_parts.push_back({"text", user_prompt});
    }
    user_msg.content_parts.push_back({"media_marker", mtmd_default_marker()});

    std::vector<common_chat_msg> messages;
    if (!system_prompt.empty()) {
        common_chat_msg system_msg;
        system_msg.role = "system";
        system_msg.content = system_prompt;
        messages.push_back(std::move(system_msg));
    }
    messages.push_back(std::move(user_msg));

    common_chat_templates_inputs inputs;
    inputs.messages = std::move(messages);
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking = false;

    common_chat_params chat_params;
    try {
        chat_params = common_chat_templates_apply(handle->tmpls.get(), inputs);
    } catch (const std::exception & e) {
        mtmd_bitmap_free(bitmap);
        return set_error(handle, std::string("failed to apply chat template: ") + e.what());
    }

    mtmd_input_text text{};
    text.text = chat_params.prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        mtmd_bitmap_free(bitmap);
        return set_error(handle, "failed to allocate mtmd chunks");
    }

    const mtmd_bitmap * bitmaps[] = {bitmap};
    if (mtmd_tokenize(handle->mctx, chunks, &text, bitmaps, 1) != 0) {
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return set_error(handle, "mtmd tokenization failed");
    }

    llama_set_embeddings(handle->lctx, false);
    llama_memory_clear(llama_get_memory(handle->lctx), true);

    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(handle->mctx, handle->lctx, chunks, 0, 0, handle->n_batch, true, &n_past) != 0) {
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return set_error(handle, "mtmd evaluation failed");
    }

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);

    common_params_sampling sampling;
    sampling.top_k = 64;
    sampling.top_p = top_p > 0.0f ? top_p : 1.0f;
    sampling.temp = temperature;

    std::string response_grammar = chat_params.grammar;
    if (!json_schema.empty()) {
        try {
            response_grammar = json_schema_to_grammar(nlohmann::ordered_json::parse(json_schema));
        } catch (const std::exception & e) {
            return set_error(handle, std::string("failed to convert json schema to grammar: ") + e.what());
        }
    }

    if (!response_grammar.empty()) {
        const auto grammar_type = json_schema.empty() ? COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT : COMMON_GRAMMAR_TYPE_USER;
        sampling.grammar = {grammar_type, response_grammar};
        sampling.grammar_lazy = json_schema.empty() ? chat_params.grammar_lazy : false;
        sampling.grammar_triggers = json_schema.empty() ? chat_params.grammar_triggers : std::vector<common_grammar_trigger>{};
        sampling.generation_prompt = json_schema.empty() ? chat_params.generation_prompt : std::string();
    }

    common_sampler_ptr sampler;
    try {
        sampler.reset(common_sampler_init(handle->model, sampling));
    } catch (const std::exception & e) {
        return set_error(handle, std::string("failed to initialize sampler: ") + e.what());
    }

    llama_batch batch = llama_batch_init(1, 0, 1);
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(static_cast<size_t>(max_tokens));

    for (int32_t i = 0; i < max_tokens; i++) {
        const llama_token token_id = common_sampler_sample(sampler.get(), handle->lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(sampler.get(), token_id, true);

        if (llama_vocab_is_eog(llama_model_get_vocab(handle->model), token_id)) {
            break;
        }

        common_batch_clear(batch);
        common_batch_add(batch, token_id, n_past++, {0}, true);
        if (llama_decode(handle->lctx, batch) != 0) {
            llama_batch_free(batch);
            return set_error(handle, "failed to decode generated token");
        }
    }

    llama_batch_free(batch);

    std::string generated = common_detokenize(handle->lctx, generated_tokens, true);
    const auto nul_pos = generated.find('\0');
    if (nul_pos != std::string::npos) {
        generated.resize(nul_pos);
    }
    while (!generated.empty() && (generated.back() == '\n' || generated.back() == '\r' || generated.back() == ' ' || generated.back() == '\t')) {
        generated.pop_back();
    }

    if (static_cast<int32_t>(generated.size()) >= out_len) {
        return set_error(handle, "generation output exceeded buffer");
    }

    std::memcpy(out, generated.data(), generated.size());
    out[generated.size()] = '\0';
    return 0;
}

} // namespace

imgsearch_llama_handle * imgsearch_llama_new(
    const char * model_path,
    const char * mmproj_path,
    int32_t n_gpu_layers,
    int32_t n_ctx,
    int32_t n_batch,
    int32_t n_threads,
    int32_t use_gpu,
    int32_t image_max_side,
    int32_t image_max_tokens,
    int32_t flash_attn_type,
    int32_t cache_type_k,
    int32_t cache_type_v) {
    const std::string model = trim(model_path);
    const std::string mmproj = trim(mmproj_path);
    if (model.empty()) {
        set_global_error("model_path is required");
        return nullptr;
    }
    if (mmproj.empty()) {
        set_global_error("mmproj_path is required");
        return nullptr;
    }

    std::call_once(g_backend_once, []() {
        load_backends();
        llama_backend_init();
        llama_log_set(native_log_callback, nullptr);
        mtmd_helper_log_set(native_log_callback, nullptr);
    });

    std::unique_ptr<imgsearch_llama_handle> handle(new imgsearch_llama_handle());

    llama_model_params mparams = llama_model_default_params();
    if (n_gpu_layers >= 0) {
        mparams.n_gpu_layers = n_gpu_layers;
    }

    handle->model = llama_model_load_from_file(model.c_str(), mparams);
    if (handle->model == nullptr) {
        set_global_error("failed to load llama.cpp model");
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? static_cast<uint32_t>(n_ctx) : 8192;
    cparams.n_batch = n_batch > 0 ? static_cast<uint32_t>(n_batch) : 512;
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_seq_max = 1;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.embeddings = true;
    if (n_threads > 0) {
        cparams.n_threads = n_threads;
        cparams.n_threads_batch = n_threads;
    }
    if (flash_attn_type >= 0) {
        cparams.flash_attn_type = static_cast<llama_flash_attn_type>(flash_attn_type);
    }
    if (cache_type_k >= 0) {
        cparams.type_k = static_cast<ggml_type>(cache_type_k);
    }
    if (cache_type_v >= 0) {
        cparams.type_v = static_cast<ggml_type>(cache_type_v);
    }

    handle->lctx = llama_init_from_model(handle->model, cparams);
    if (handle->lctx == nullptr) {
        llama_model_free(handle->model);
        handle->model = nullptr;
        set_global_error("failed to initialize llama.cpp context");
        return nullptr;
    }

    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu = use_gpu != 0;
    vparams.n_threads = n_threads > 0 ? n_threads : 0;
    vparams.print_timings = false;
    vparams.media_marker = mtmd_default_marker();
    vparams.warmup = true;
    if (image_max_tokens > 0) {
        vparams.image_max_tokens = image_max_tokens;
    }
    if (flash_attn_type >= 0) {
        vparams.flash_attn_type = static_cast<llama_flash_attn_type>(flash_attn_type);
    }

    handle->mctx = mtmd_init_from_file(mmproj.c_str(), handle->model, vparams);
    if (handle->mctx == nullptr) {
        llama_free(handle->lctx);
        handle->lctx = nullptr;
        llama_model_free(handle->model);
        handle->model = nullptr;
        set_global_error("failed to initialize mtmd context");
        return nullptr;
    }

    handle->dims = llama_model_n_embd_out(handle->model);
    if (handle->dims <= 0) {
        mtmd_free(handle->mctx);
        handle->mctx = nullptr;
        llama_free(handle->lctx);
        handle->lctx = nullptr;
        llama_model_free(handle->model);
        handle->model = nullptr;
        set_global_error("invalid llama.cpp embedding dimension");
        return nullptr;
    }

    handle->n_batch = n_batch > 0 ? n_batch : 512;
    handle->image_max_side = image_max_side > 0 ? image_max_side : kDefaultImageMaxSide;
    handle->last_error.clear();
    return handle.release();
}

void imgsearch_llama_free(imgsearch_llama_handle * handle) {
    if (handle == nullptr) {
        return;
    }
    if (handle->mctx != nullptr) {
        mtmd_free(handle->mctx);
        handle->mctx = nullptr;
    }
    if (handle->lctx != nullptr) {
        llama_free(handle->lctx);
        handle->lctx = nullptr;
    }
    if (handle->model != nullptr) {
        llama_model_free(handle->model);
        handle->model = nullptr;
    }
    delete handle;
}

int32_t imgsearch_llama_dims(const imgsearch_llama_handle * handle) {
    if (handle == nullptr) {
        return 0;
    }
    return handle->dims;
}

int32_t imgsearch_llama_embed_text(
    imgsearch_llama_handle * handle,
    const char * text,
    const char * instruction,
    float * out,
    int32_t out_len) {
    const std::string prompt = compose_prompt(instruction, text);
    if (prompt.empty()) {
        return set_error(handle, "text embedding prompt is empty");
    }
    return eval_prompt(handle, prompt, nullptr, 0, 0, out, out_len);
}

int32_t imgsearch_llama_embed_image(
    imgsearch_llama_handle * handle,
    const char * image_path,
    const char * instruction,
    float * out,
    int32_t out_len) {
    if (handle == nullptr) {
        return set_error(handle, "llama.cpp handle is not initialized");
    }

    const std::string image = trim(image_path);
    if (image.empty()) {
        return set_error(handle, "image_path is required");
    }

    mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_file(handle->mctx, image.c_str());
    if (bitmap == nullptr) {
        return set_error(handle, "failed to decode image for mtmd input");
    }

    bitmap = maybe_resize_bitmap(handle, bitmap);
    if (bitmap == nullptr) {
        return -1;
    }

    const std::string prompt = compose_prompt(instruction, mtmd_default_marker());
    const mtmd_bitmap * bitmaps[] = {bitmap};
    int32_t res = eval_prompt(handle, prompt, bitmaps, 1, 0, out, out_len);

    mtmd_bitmap_free(bitmap);
    return res;
}

int32_t imgsearch_llama_inspect_embed_image(
    imgsearch_llama_handle * handle,
    const char * image_path,
    const char * instruction,
    imgsearch_llama_embed_inspect * out) {
    const std::string image = trim(image_path);
    if (image.empty()) {
        return set_error(handle, "image_path is required");
    }

    mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_file(handle->mctx, image.c_str());
    if (bitmap == nullptr) {
        return set_error(handle, "failed to decode image for mtmd input");
    }

    bitmap = maybe_resize_bitmap(handle, bitmap);
    if (bitmap == nullptr) {
        return -1;
    }

    const std::string prompt = compose_prompt(instruction, mtmd_default_marker());
    const mtmd_bitmap * bitmaps[] = {bitmap};
    int32_t res = inspect_prompt(handle, prompt, bitmaps, 1, out);

    mtmd_bitmap_free(bitmap);
    return res;
}

int32_t imgsearch_llama_generate_image(
    imgsearch_llama_handle * handle,
    const char * image_path,
    const char * system_prompt,
    const char * user_prompt,
    const char * json_schema,
    int32_t max_tokens,
    float temperature,
    float top_p,
    char * out,
    int32_t out_len) {
    return generate_image(
        handle,
        trim(image_path),
        trim(system_prompt),
        trim(user_prompt),
        trim(json_schema),
        max_tokens,
        temperature,
        top_p,
        out,
        out_len);
}

const char * imgsearch_llama_last_error(const imgsearch_llama_handle * handle) {
    if (handle == nullptr) {
        return "";
    }
    return handle->last_error.c_str();
}

const char * imgsearch_llama_global_error(void) {
    std::lock_guard<std::mutex> lock(g_error_mu);
    return g_last_error.c_str();
}

#else

#include <string>

namespace {
std::string g_last_error = "llama-cpp-native requires build tag 'llamacpp_native'";
}

imgsearch_llama_handle * imgsearch_llama_new(
    const char *,
    const char *,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t) {
    return nullptr;
}

void imgsearch_llama_free(imgsearch_llama_handle *) {}

int32_t imgsearch_llama_dims(const imgsearch_llama_handle *) {
    return 0;
}

int32_t imgsearch_llama_embed_text(
    imgsearch_llama_handle *,
    const char *,
    const char *,
    float *,
    int32_t) {
    return -1;
}

int32_t imgsearch_llama_embed_image(
    imgsearch_llama_handle *,
    const char *,
    const char *,
    float *,
    int32_t) {
    return -1;
}

int32_t imgsearch_llama_inspect_embed_image(
    imgsearch_llama_handle *,
    const char *,
    const char *,
    imgsearch_llama_embed_inspect *) {
    return -1;
}

int32_t imgsearch_llama_generate_image(
    imgsearch_llama_handle *,
    const char *,
    const char *,
    const char *,
    const char *,
    int32_t,
    float,
    float,
    char *,
    int32_t) {
    return -1;
}

const char * imgsearch_llama_last_error(const imgsearch_llama_handle *) {
    return g_last_error.c_str();
}

const char * imgsearch_llama_global_error(void) {
    return g_last_error.c_str();
}

#endif
