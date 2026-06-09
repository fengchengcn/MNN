// Created by ruoyi.sjd on 2025/01/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.os.Build
import android.util.Log
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import org.json.JSONObject
import java.io.File

/**
 * Utility class for getting voice model paths dynamically based on MainSettings defaults
 */
object VoiceModelPathUtils {
    private const val TAG = "VoiceModelPathUtils"
    const val DEFAULT_TTS_SAMPLE_RATE = 44100
    
    // Fallback paths when no default model is set
    private const val FALLBACK_ASR_MODEL_DIR = "/data/local/tmp/asr_models"
    private const val FALLBACK_TTS_MODEL_DIR = "/data/local/tmp/test_new_tts/bert-vits/"
    private const val LOCAL_MODELS_DIR = "/data/local/tmp/mnn_models"

    /**
     * Check if a model directory is an Omni audio model (new llmexport.py path).
     * Identified by audio.mnn + config.json with is_audio=true.
     */
    private fun isOmniAudioModel(dir: File): Boolean {
        if (!File(dir, "audio.mnn").exists() || !File(dir, "config.json").exists()) {
            return false
        }
        return try {
            val config = JSONObject(File(dir, "config.json").readText())
            val isAudio = config.optBoolean("is_audio", false)
            if (isAudio) {
                Log.d(TAG, "Detected Omni audio model at: ${dir.absolutePath}")
            }
            isAudio
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse config.json for Omni audio detection: ${e.message}")
            false
        }
    }

    /**
     * Scan /data/local/tmp/mnn_models/ for a Qwen3-ASR model directory.
     * Detects both:
     * - Old path: audio_encoder.mnn presence
     * - New Omni path: audio.mnn + config.json with is_audio=true
     * @return the first matching directory path, or null if none found.
     */
    private fun findQwen3AsrInLocalModels(): String? {
        val localDir = File(LOCAL_MODELS_DIR)
        if (!localDir.exists() || !localDir.isDirectory) return null
        localDir.listFiles()?.forEach { subdir ->
            if (!subdir.isDirectory) return@forEach
            // Old path: check for audio_encoder.mnn
            if (File(subdir, "audio_encoder.mnn").exists()) {
                Log.i(TAG, "Found Qwen3-ASR model (old) in local models: ${subdir.absolutePath}")
                return subdir.absolutePath
            }
            // New Omni path: check for audio.mnn + config.json is_audio=true
            if (isOmniAudioModel(subdir)) {
                Log.i(TAG, "Found Qwen3-ASR model (Omni) in local models: ${subdir.absolutePath}")
                return subdir.absolutePath
            }
        }
        return null
    }

    /**
     * Get the ASR model directory path based on the default ASR model setting
     * @param context Android context
     * @return ASR model directory path, or fallback path if no default model is set
     */
    fun getAsrModelPath(context: Context): String {
        val defaultAsrModel = MainSettings.getDefaultAsrModel(context)
        Log.d(TAG, "Getting ASR model path for default model: $defaultAsrModel")

        if (defaultAsrModel.isNullOrEmpty()) {
            // Scan mnn_models for Qwen3-ASR first
            val qwen3Path = findQwen3AsrInLocalModels()
            if (qwen3Path != null) {
                Log.i(TAG, "Using Qwen3-ASR from local models: $qwen3Path")
                return qwen3Path
            }
            // Check legacy asr_models path
            if (File(FALLBACK_ASR_MODEL_DIR, "audio_encoder.mnn").exists() ||
                File(FALLBACK_ASR_MODEL_DIR).exists()) {
                Log.w(TAG, "No default ASR model set, using fallback path: $FALLBACK_ASR_MODEL_DIR")
                return FALLBACK_ASR_MODEL_DIR
            }
            // Last resort
            Log.w(TAG, "No ASR model found, returning fallback: $FALLBACK_ASR_MODEL_DIR")
            return FALLBACK_ASR_MODEL_DIR
        }
        
        try {
            val modelDownloadManager = ModelDownloadManager.getInstance(context)
            val asrModelFile = modelDownloadManager.getDownloadedFile(defaultAsrModel)
            
            if (asrModelFile != null && asrModelFile.exists()) {
                val modelPath = asrModelFile.absolutePath
                Log.i(TAG, "Found ASR model path: $modelPath for model: $defaultAsrModel")
                return modelPath
            } else {
                Log.w(TAG, "ASR model file not found for: $defaultAsrModel, using fallback path: $FALLBACK_ASR_MODEL_DIR")
                return FALLBACK_ASR_MODEL_DIR
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting ASR model path for: $defaultAsrModel", e)
            return FALLBACK_ASR_MODEL_DIR
        }
    }
    
    /**
     * Get the TTS model directory path based on the default TTS model setting
     * @param context Android context
     * @return TTS model directory path, or fallback path if no default model is set
     */
    fun getTtsModelPath(context: Context): String {
        val defaultTtsModel = MainSettings.getDefaultTtsModel(context)
        Log.d(TAG, "Getting TTS model path for default model: $defaultTtsModel")
        
        if (defaultTtsModel.isNullOrEmpty()) {
            Log.w(TAG, "No default TTS model set, using fallback path: $FALLBACK_TTS_MODEL_DIR")
            return FALLBACK_TTS_MODEL_DIR
        }
        
        try {
            val modelDownloadManager = ModelDownloadManager.getInstance(context)
            val ttsModelFile = modelDownloadManager.getDownloadedFile(defaultTtsModel)
            
            if (ttsModelFile != null && ttsModelFile.exists()) {
                val modelPath = ttsModelFile.absolutePath
                Log.i(TAG, "Found TTS model path: $modelPath for model: $defaultTtsModel")
                return modelPath
            } else {
                Log.w(TAG, "TTS model file not found for: $defaultTtsModel, using fallback path: $FALLBACK_TTS_MODEL_DIR")
                return FALLBACK_TTS_MODEL_DIR
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting TTS model path for: $defaultTtsModel", e)
            return FALLBACK_TTS_MODEL_DIR
        }
    }

    fun getTtsSampleRate(modelDir: String?, fallbackSampleRate: Int = DEFAULT_TTS_SAMPLE_RATE): Int {
        if (modelDir.isNullOrBlank()) {
            Log.w(TAG, "TTS model path is empty, using fallback sample rate: $fallbackSampleRate")
            return fallbackSampleRate
        }

        val configFile = File(modelDir, "config.json")
        if (!configFile.exists() || !configFile.isFile) {
            Log.w(TAG, "TTS config.json not found at ${configFile.absolutePath}, using fallback sample rate: $fallbackSampleRate")
            return fallbackSampleRate
        }

        return try {
            val configJson = JSONObject(configFile.readText())
            val sampleRate = when (val value = configJson.opt("sample_rate")) {
                is Number -> value.toInt()
                is String -> value.toIntOrNull()
                else -> null
            }

            if (sampleRate != null && sampleRate > 0) {
                Log.i(TAG, "Loaded TTS sample rate: $sampleRate from ${configFile.absolutePath}")
                sampleRate
            } else {
                Log.w(TAG, "Invalid TTS sample rate in ${configFile.absolutePath}, using fallback: $fallbackSampleRate")
                fallbackSampleRate
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse TTS sample rate from ${configFile.absolutePath}", e)
            fallbackSampleRate
        }
    }

    fun getTtsLanguage(context: Context): String {
        val config = context.resources.configuration
        val locale = if (Build.VERSION.SDK_INT >= 24) {
            if (config.locales.isEmpty) null else config.locales[0]
        } else {
            @Suppress("DEPRECATION")
            config.locale
        }

        return if (locale?.language == "zh") "zh" else "en"
    }
    
    /**
     * Check if voice models are ready and get their status
     * @param context Android context
     * @return Pair<Boolean, String> where first is readiness status and second is status message
     */
    fun checkVoiceModelsStatus(context: Context): Pair<Boolean, String> {
        val defaultTtsModel = MainSettings.getDefaultTtsModel(context)
        val defaultAsrModel = MainSettings.getDefaultAsrModel(context)
        
        val statusBuilder = StringBuilder()
        
        if (defaultTtsModel.isNullOrEmpty()) {
            statusBuilder.append("No default TTS model set. ")
        }
        if (defaultAsrModel.isNullOrEmpty()) {
            statusBuilder.append("No default ASR model set. ")
        }
        
        if (statusBuilder.isNotEmpty()) {
            Log.w(TAG, "Voice models not ready: $statusBuilder")
            return false to statusBuilder.toString()
        }
        
        try {
            val modelDownloadManager = ModelDownloadManager.getInstance(context)
            
            val ttsModelFile = modelDownloadManager.getDownloadedFile(defaultTtsModel!!)
            val asrModelFile = modelDownloadManager.getDownloadedFile(defaultAsrModel!!)
            
            if (ttsModelFile == null || !ttsModelFile.exists()) {
                statusBuilder.append("TTS model file not found: $defaultTtsModel. ")
            }
            if (asrModelFile == null || !asrModelFile.exists()) {
                statusBuilder.append("ASR model file not found: $defaultAsrModel. ")
            }
            
            if (statusBuilder.isNotEmpty()) {
                Log.w(TAG, "Voice models not ready: $statusBuilder")
                return false to statusBuilder.toString()
            }
            
            Log.i(TAG, "Voice models ready - TTS: $defaultTtsModel, ASR: $defaultAsrModel")
            return true to "Voice models ready"
            
        } catch (e: Exception) {
            val errorMsg = "Error checking voice models: ${e.message}"
            Log.e(TAG, errorMsg, e)
            return false to errorMsg
        }
    }
} 
