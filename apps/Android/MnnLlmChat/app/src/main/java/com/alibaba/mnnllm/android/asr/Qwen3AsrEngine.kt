package com.alibaba.mnnllm.android.asr

import android.util.Log

/**
 * Qwen3-ASR inference engine — Kotlin wrapper for native JNI.
 *
 * Usage:
 *   val engine = Qwen3AsrEngine()
 *   engine.init("/path/to/model/dir", numThreads = 4)
 *   engine.pushAudio(pcmFloats)     // feed 16kHz mono PCM as FloatArray
 *   engine.endAudio()               // signal end, run decoder
 *   val text = engine.getResult()   // get transcription
 *   engine.reset()                  // ready for next utterance
 *   engine.release()                // cleanup
 */
class Qwen3AsrEngine {
    companion object {
        private const val TAG = "Qwen3AsrEngine"

        init {
            System.loadLibrary("mnnllmapp")
        }
    }

    // Native pointer (set by nativeInit)
    @Volatile
    private var mNativePtr: Long = 0

    /**
     * Initialize the ASR engine.
     * @param modelDir Path to directory containing one of:
     *   New format (llmexport.py): audio.mnn, llm.mnn, llm.mnn.weight, embeddings_bf16.bin, tokenizer.txt
     *   Legacy format (sherpa-onnx): audio_encoder.mnn, llm_kv_8bit.mnn, llm_kv_8bit.mnn.weight, embeddings_bf16.bin
     * @param cacheDir App's internal cache directory for temp files (must be writable)
     * @param numThreads Number of inference threads (default: 4, Phase 3: raised from 2)
     * @return true if initialization succeeded
     */
    fun init(modelDir: String, cacheDir: String, numThreads: Int = 4): Boolean {
        Log.i(TAG, "init: modelDir=$modelDir, cacheDir=$cacheDir, numThreads=$numThreads")
        val result = nativeInit(modelDir, cacheDir, numThreads)
        Log.i(TAG, "init result: $result")
        return result
    }

    /**
     * Push PCM audio samples.
     * @param pcmData 16kHz mono PCM as normalized FloatArray (range [-1.0, 1.0])
     */
    fun pushAudio(pcmData: FloatArray) {
        nativePushAudio(pcmData)
    }

    /**
     * Signal end of audio. Triggers decoder inference.
     * For streaming mode, use {@link #startDecode()} instead.
     */
    fun endAudio() {
        nativeEndAudio()
    }

    /**
     * Phase 2 Streaming: Start incremental decode.
     * Runs audio encoder + prefill. Returns true if first token is ready.
     * After this, call {@link #decodeStep()} in a loop for each subsequent token.
     */
    fun startDecode(): Boolean {
        val ok = nativeStartDecode()
        Log.i(TAG, "startDecode: $ok")
        return ok
    }

    /**
     * Phase 2 Streaming: Single decode step.
     * @return Pair(hasMoreTokens, tokenId) — hasMoreTokens=false when EOS/max reached.
     */
    fun decodeStep(): Pair<Boolean, Int> {
        val tokenOut = IntArray(1)
        val more = nativeDecodeStep(tokenOut)
        return Pair(more, tokenOut[0])
    }

    /**
     * Phase 2 Streaming: Check if incremental decode is active.
     * Returns true between startDecode() and decode loop completion.
     */
    fun isDecoding(): Boolean {
        return nativeIsDecoding()
    }

    /**
     * Phase 2 Streaming: Get partial result text during decode (non-blocking).
     * Returns the current transcription based on tokens generated so far.
     * For the final result after decode completes, use {@link #getResultText()}.
     */
    fun getPartialResult(): String {
        return nativeGetPartialResult()
    }

    /**
     * Get transcription result as space-separated token IDs.
     * Raw token IDs for custom decoding.
     */
    fun getResult(): String {
        return nativeGetResult()
    }

    /**
     * Get decoded text transcription.
     * Uses tokenizer.txt lookup (basic decoding — for proper text,
     * use the token IDs with a full BPE tokenizer on the app side).
     */
    fun getResultText(): String {
        return nativeGetResultText()
    }

    /**
     * Reset engine state for a new utterance.
     * Models stay loaded — only audio buffer and decoder state are cleared.
     */
    fun reset() {
        nativeReset()
    }

    /**
     * Release all native resources.
     * Call when done — engine cannot be used after this.
     */
    fun release() {
        nativeRelease()
    }

    // JNI declarations
    private external fun nativeInit(modelDir: String, cacheDir: String, numThreads: Int): Boolean
    private external fun nativePushAudio(pcmData: FloatArray)
    private external fun nativeEndAudio()
    private external fun nativeGetResult(): String
    private external fun nativeGetResultText(): String
    private external fun nativeReset()
    private external fun nativeRelease()

    // Phase 2 Streaming JNI
    private external fun nativeStartDecode(): Boolean
    private external fun nativeDecodeStep(tokenOut: IntArray): Boolean
    private external fun nativeIsDecoding(): Boolean
    private external fun nativeGetPartialResult(): String
}
