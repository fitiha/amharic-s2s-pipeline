# Amharic Voice Assistant for Banking

A low-latency, fully open-source Amharic speech-to-speech assistant that answers customer questions from a bank's FAQ and product documents. Speak a question in Amharic, get a spoken Amharic answer grounded in the documents, with a polite refusal when the answer isn't in them.

```
mic -> ASR -> hybrid RAG retrieval -> LLM (streamed) -> TTS -> spoken answer
        |            |                     |              |
   Wav2Vec2 CTC   BGE-M3 + BM25      Gemma 4 E4B      Amharic OmniVoice
   (fp16)         (FAISS, RRF)       (4-bit, llama.cpp)  (8 diffusion steps)
```

## Why this exists

Amharic is a low-resource language, so off-the-shelf voice assistants perform poorly in it. This project shows that a usable pipeline can be built from open models on two T4 GPUs, and records which optimizations mattered and which experiments failed.

## Results

| What | Result |
|---|---|
| Time to first audio (end to end) | **9.02 s -> 1.2 s (85% lower)** after the optimizations below |
| LLM throughput | **18.6 -> 53 tok/s (2.8x)** |
| ASR | 0.41 s for 5 s of audio (T4, fp16) |
| TTS, one short sentence | 1.34 s at 32 steps, 0.70 s at 16, **0.37 s at 8** (informal listening test: 8 steps sounded the same as 32) |
| Retrieval (embed + hybrid search) | about 43 to 53 ms per query |
| LLM prompt processing | 282 to 465 ms for 338 to 624 prompt tokens |
| Index | 41 documents -> 1,812 chunks, built in 6.2 s |

Timings are measured on Kaggle (2x Tesla T4). The "time to first audio" figure is an estimate that sums ASR, first LLM clause and first TTS chunk. It does not include endpoint waiting, browser upload or playback buffering.

## What made it fast

- **Streaming and clause chunking.** The LLM response is streamed. TTS starts on the first complete sentence instead of waiting for the whole answer.
- **4-bit quantization.** The model runs as `Q4_K_M` GGUF through the `llama.cpp` server with full GPU offload.
- **Multi-GPU workload isolation.** The LLM runs alone on GPU 0. ASR and TTS share GPU 1, so they don't compete for memory or compute.
- **Fewer TTS diffusion steps.** Dropping from 32 to 8 steps made speech generation about 3.6x faster.
- **Bounded prompts.** Retrieval returns at most 2 chunks of about 400 characters each, which keeps the prompt small and prompt processing fast.
- **Semantic response cache.** Exact and near-duplicate questions (cosine similarity of 0.93 or more) skip the LLM and TTS and return the stored text and audio. The cache can be pre-warmed with common questions; if you do this, say so when you report latency.

## How retrieval works

- Documents (`.docx`) are split into question-and-answer pairs and text blocks with an Amharic-aware sentence splitter (it splits on the Ethiopic full stop `።` and question mark `፧`).
- **Hybrid search:** dense retrieval (BGE-M3 embeddings in FAISS) plus BM25, combined with reciprocal rank fusion.
- **Out-of-scope guard:** if the best dense similarity is below a threshold, no context is passed to the LLM.
- The system prompt tells the model to answer only from the provided context in at most two short sentences, and to use one fixed refusal sentence otherwise.

## Models

| Stage | Model |
|---|---|
| ASR | `snapwre/hohe-asr-amharic` (Wav2Vec2 CTC) |
| Embeddings | `BAAI/bge-m3` |
| LLM | Gemma 4 E4B, `bartowski/google_gemma-4-E4B-it-GGUF` (Q4_K_M) via `llama.cpp` |
| TTS | `gheero-Leyu/amharic-omnivoice-tts` (fallback: `african-low-resource/omnivoice-amharic`) |

## Run it (Kaggle)

1. New notebook with **Accelerator: GPU T4 x2** and **Internet: ON**.
2. Add a Kaggle secret named `HF_TOKEN`. Optionally add `DEMO_PASSWORD` for the demo login.
3. Attach your own document dataset with `.docx` files in `faq/` and `general/` folders. **The bank documents used in development are not included in this repository.**
4. Add this repository's `rag/` folder as an input, then open `amharic_voice_assistant.ipynb` and run the sections in order:
   - A: build and start the `llama.cpp` server (about 30 minutes the first time; later runs reuse the build)
   - B: load ASR and TTS
   - D and E: extract documents and build the index
   - F: prompt and answer function
   - G: Gradio voice demo (open the public link in a new tab so the microphone prompt appears)
   - H (optional): OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints, intended for orchestration frameworks such as Pipecat

## What went wrong and what we changed

- **Context overflow.** A 4,096-token context caused `exceed_context_size_error`. Fixed by raising the context to 8,192 and capping retrieved context.
- **Instruction echoed as the answer.** For an out-of-scope question, the model replied with the instruction text itself ("tell the user politely that you cannot answer") instead of refusing. Fixed by giving the model an exact refusal sentence to use.
- **Bad chunking.** The first extractor made chunks only at headings and question lines. Replacing it with a hybrid chunker and an Amharic sentence splitter changed the index from 1,013 to 1,812 chunks and fixed documents with no headings.

## Limitations

- **No labelled retrieval evaluation yet.** Retrieval was checked by hand on about 10 questions, comparing the old and new retriever. There is no precision/recall measurement, no ASR word error rate, and no native-speaker rating of the answers or the voice.
- **The out-of-scope guard is weak.** The retrieval threshold is 0.35, but unrelated questions scored 0.45 to 0.48 in testing while relevant ones scored 0.56 to 0.75. So the guard rarely triggers, and refusals currently depend on the prompt. A threshold near 0.5 separates these examples, but it has not been validated on a larger set.
- **Prompt-injection robustness has not been tested in this repository.**
- **The cache threshold (0.93) has not been tuned.**
- **Latency numbers exclude network and playback time,** and were measured on one Kaggle machine.

## Next steps

- Build a labelled question set from real user queries and measure retrieval precision and answer faithfulness.
- Add ASR word error rate and native-speaker evaluation.
- Tune the out-of-scope threshold and test adversarial prompts.
- Integrate the OpenAI-compatible endpoints with a real-time orchestration framework.

## License

MIT
