// Created for WebRTC-style audio preprocessing integration.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm

/**
 * WebRTC-style audio preprocessor for ASR.
 *
 * Provides adaptive gain control (AGC), high-pass filtering, and noise gating
 * based on the WebRTC AGC2 algorithm. Designed to normalize varying microphone
 * distances before audio reaches VAD and the ASR engine.
 *
 * Usage:
 *   val pp = AudioPreprocessor()
 *   pp.process(samples)  // in-place, call per audio chunk before VAD
 *   pp.reset()           // at start of new recording
 *   pp.release()         // when done
 */
class AudioPreprocessor {

    private var nativePtr: Long = 0

    /**
     * Create the native preprocessor with default WebRTC-style config:
     *   - Target: -6 dBFS (RMS = 0.5)
     *   - Attack: 20ms (fast gain increase for quiet speakers)
     *   - Release: 200ms (slow decrease to avoid pumping)
     *   - Max gain: 20x (26 dB)
     *   - Noise gate: -50 dBFS
     *   - HPF: 80 Hz Butterworth
     */
    fun create() {
        if (nativePtr != 0L) release()
        nativePtr = nativeCreate()
    }

    /**
     * Process audio samples in-place. Normalizes levels adaptively so that
     * speech is brought to -6 dBFS regardless of microphone distance.
     * Must call [create] first.
     */
    fun process(samples: FloatArray) {
        if (nativePtr == 0L) return
        nativeProcess(nativePtr, samples)
    }

    /** Reset internal state for a new recording session. */
    fun reset() {
        if (nativePtr == 0L) return
        nativeReset(nativePtr)
    }

    /** Current smoothed gain (for diagnostics/logging). */
    fun getGain(): Float {
        if (nativePtr == 0L) return 1.0f
        return nativeGetGain(nativePtr)
    }

    /** Release native resources. */
    fun release() {
        if (nativePtr == 0L) return
        nativeRelease(nativePtr)
        nativePtr = 0L
    }

    protected fun finalize() {
        release()
    }

    // ── JNI ──
    private external fun nativeCreate(): Long
    private external fun nativeProcess(ptr: Long, samples: FloatArray)
    private external fun nativeReset(ptr: Long)
    private external fun nativeRelease(ptr: Long)
    private external fun nativeGetGain(ptr: Long): Float
}
