package com.alibaba.mnnllm.multimodal.audio

import android.content.Context
import android.util.Log
import java.io.File

class AudioHandler(private val context: Context) {
    private var simpleWaveRecorder: SimpleWaveRecorder? = null
    private var currentRecordingPath: String? = null

    fun startRecording() {
        val fileName = "record_${System.currentTimeMillis()}.wav"
        val file = File(context.cacheDir, fileName)
        currentRecordingPath = file.absolutePath
        
        simpleWaveRecorder = SimpleWaveRecorder()
        simpleWaveRecorder?.startRecording(currentRecordingPath!!)
        
        Log.d("AudioHandler", "Started recording to $currentRecordingPath")
    }

    fun stopRecording(): String? {
        simpleWaveRecorder?.stopRecording()
        val path = currentRecordingPath
        Log.d("AudioHandler", "Stopped recording, path: $path")
        simpleWaveRecorder = null
        currentRecordingPath = null
        return path
    }
}
