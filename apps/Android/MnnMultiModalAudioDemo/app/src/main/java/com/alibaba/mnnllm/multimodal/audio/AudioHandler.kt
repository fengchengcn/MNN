package com.alibaba.mnnllm.multimodal.audio

import android.content.Context
import android.util.Log
import java.io.File

class AudioHandler(private val context: Context) {
    private var simpleWaveRecorder: SimpleWaveRecorder? = null
    private var currentRecordingPath: String? = null
    var onAmplitudeUpdate: ((Int) -> Unit)? = null

    fun startRecording() {
        val fileName = "record_${System.currentTimeMillis()}.wav"
        val file = File(context.cacheDir, fileName)
        currentRecordingPath = file.absolutePath

        simpleWaveRecorder = SimpleWaveRecorder()
        simpleWaveRecorder?.onAmplitudeUpdate = onAmplitudeUpdate
        simpleWaveRecorder?.startRecording(currentRecordingPath!!)

        Log.d("AudioHandler", "Started recording to $currentRecordingPath")
    }

    fun stopRecording(): String? {
        simpleWaveRecorder?.stopRecording()
        val path = currentRecordingPath
        if (path != null) {
            val file = File(path)
            Log.d("AudioHandler", "Stopped recording, path: $path, size: ${file.length()} bytes")
        }
        simpleWaveRecorder = null
        currentRecordingPath = null
        return path
    }
}
