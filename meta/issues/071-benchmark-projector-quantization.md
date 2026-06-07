# Benchmark projector quantization for ingestion

## Summary

The Gemma and Qwen multimodal projectors are stored as f16 GGUF files. Quantizing projector weights may reduce CPU memory bandwidth, RAM/VRAM footprint, and ingestion latency, but it must be tested for quality and runtime compatibility.

## Requirements

- Produce quantized projector variants, starting with conservative Q8_0 where supported.
- Benchmark Gemma annotator projector quantization and Qwen embedding projector quantization separately.
- Run quick per-model benchmarks and full ingestion benchmarks for promising variants.
- Test both CPU-only and CUDA configurations.
- Review annotation quality and embedding/search behavior for regressions.

## Acceptance Criteria

- At least one quantized Gemma mmproj and one quantized Qwen mmproj variant are benchmarked or a concrete compatibility blocker is documented.
- Results include timing, RAM/VRAM footprint where available, and quality notes.
- The repository documents whether projector quantization should be adopted, rejected, or kept as an optional configuration.

## Notes

- Q8_0 is the first target because it is less likely to harm vision-language alignment than lower-bit quantization.
- Lower-bit projector quantization should only be tried after Q8_0 establishes whether projector bandwidth is a meaningful bottleneck.
