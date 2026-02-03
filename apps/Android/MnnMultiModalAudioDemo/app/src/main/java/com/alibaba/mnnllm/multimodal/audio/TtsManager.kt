package com.alibaba.mnnllm.multimodal.audio

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

class TtsManager(context: Context) : TextToSpeech.OnInitListener {
    private var tts: TextToSpeech? = TextToSpeech(context, this)
    private var isInitialized = false

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.CHINESE)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TtsManager", "Language not supported")
            } else {
                isInitialized = true
            }
        } else {
            Log.e("TtsManager", "Initialization failed")
        }
    }

    fun speak(text: String) {
        if (isInitialized) {
            tts?.speak(text, TextToSpeech.QUEUE_ADD, null, null)
        }
    }

    fun stop() {
        tts?.stop()
    }

    fun release() {
        tts?.stop()
        tts?.shutdown()
    }
}
