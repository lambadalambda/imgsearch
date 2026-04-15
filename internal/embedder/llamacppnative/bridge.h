#ifndef IMGSEARCH_LLAMA_BRIDGE_H
#define IMGSEARCH_LLAMA_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct imgsearch_llama_handle imgsearch_llama_handle;

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
    int32_t cache_type_v);

void imgsearch_llama_free(imgsearch_llama_handle * handle);

int32_t imgsearch_llama_dims(const imgsearch_llama_handle * handle);

int32_t imgsearch_llama_embed_text(
    imgsearch_llama_handle * handle,
    const char * text,
    const char * instruction,
    float * out,
    int32_t out_len);

int32_t imgsearch_llama_embed_image(
    imgsearch_llama_handle * handle,
    const char * image_path,
    const char * instruction,
    float * out,
    int32_t out_len);

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
    int32_t out_len);

const char * imgsearch_llama_last_error(const imgsearch_llama_handle * handle);
const char * imgsearch_llama_global_error(void);

#ifdef __cplusplus
}
#endif

#endif
