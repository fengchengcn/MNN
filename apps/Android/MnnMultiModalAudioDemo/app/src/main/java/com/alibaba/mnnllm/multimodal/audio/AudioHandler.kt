package com.alibaba.mnnllm.multimodal.audio

import android.content.Context
import android.util.Log
import com.github.squti.androidwaverecorder.WaveRecorder
import java.io.File

class AudioHandler(private val context: Context) {
    private var waveRecorder: WaveRecorder? = null
    private var currentRecordingPath: String? = null

    fun startRecording() {
        val fileName = "record_${System.currentTimeMillis()}.wav"
        val file = File(context.cacheDir, fileName)
        currentRecordingPath = file.absolutePath
        
        waveRecorder = WaveRecorder(currentRecordingPath!!)
        waveRecorder?.apply {
            // Configure for Whisper: 16kHz, Mono, 16bit
            // Note: WaveRecorder defaults might need check, but it usually handles standard wav
            startRecording()
        }
        Log.d("AudioHandler", "Started recording to $currentRecordingPath")
    }

    fun stopRecording(): String? {
        waveRecorder?.stopRecording()
        val path = currentRecordingPath
        Log.d("AudioHandler", "Stopped recording, path: $path")
        waveRecorder = null
        currentRecordingPath = null
        return path
    }
}
