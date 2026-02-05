package com.alibaba.mnnllm.multimodal.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class SimpleWaveRecorder {
    private val TAG = "SimpleWaveRecorder"

    // Audio configuration for Whisper/ASR: 16kHz, Mono, 16bit
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingJob: Job? = null
    private val coroutineScope = CoroutineScope(Dispatchers.IO)

    var onAmplitudeUpdate: ((Int) -> Unit)? = null

    @SuppressLint("MissingPermission")
    fun startRecording(outputFilePath: String) {
        if (isRecording) return

        val file = File(outputFilePath)
        audioRecord =
                AudioRecord(
                        MediaRecorder.AudioSource.VOICE_RECOGNITION,
                        sampleRate,
                        channelConfig,
                        audioFormat,
                        bufferSize
                )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord initialization failed")
            return
        }

        isRecording = true
        audioRecord?.startRecording()

        recordingJob = coroutineScope.launch { writeAudioDataToFile(file) }
    }

    fun stopRecording() {
        isRecording = false
        recordingJob?.cancel()
        recordingJob = null

        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recorder: ${e.message}")
        }
        audioRecord = null
    }

    private fun writeAudioDataToFile(file: File) {
        val shortBuffer = ShortArray(bufferSize / 2)
        FileOutputStream(file).use { out ->
            // Write placeholder for WAV header (44 bytes)
            out.write(ByteArray(44))

            while (isRecording) {
                val read = audioRecord?.read(shortBuffer, 0, shortBuffer.size) ?: 0
                if (read > 0) {
                    var maxAmplitude = 0
                    val byteBuffer =
                            java.nio.ByteBuffer.allocate(read * 2)
                                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    for (i in 0 until read) {
                        val s = shortBuffer[i]
                        // Apply software gain (e.g. 5x) to boost generic low volume
                        val amplified =
                                (s * 1.2f)
                                        .coerceIn(
                                                Short.MIN_VALUE.toFloat(),
                                                Short.MAX_VALUE.toFloat()
                                        )
                                        .toInt()
                                        .toShort()

                        val finalShort = amplified
                        val abs = Math.abs(finalShort.toInt())
                        if (abs > maxAmplitude) maxAmplitude = abs
                        byteBuffer.putShort(finalShort)
                    }
                    onAmplitudeUpdate?.invoke(maxAmplitude)
                    out.write(byteBuffer.array(), 0, read * 2)
                }
            }
        }
        // After recording, update the WAV header with actual data size
        updateWavHeader(file)
    }

    private fun updateWavHeader(file: File) {
        val totalAudioLen = file.length() - 44
        val totalDataLen = totalAudioLen + 36
        val channels = 1
        val byteRate = 16 * sampleRate * channels / 8

        val header = ByteArray(44)
        header[0] = 'R'.toByte()
        header[1] = 'I'.toByte()
        header[2] = 'F'.toByte()
        header[3] = 'F'.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = (totalDataLen shr 8 and 0xff).toByte()
        header[6] = (totalDataLen shr 16 and 0xff).toByte()
        header[7] = (totalDataLen shr 24 and 0xff).toByte()
        header[8] = 'W'.toByte()
        header[9] = 'A'.toByte()
        header[10] = 'V'.toByte()
        header[11] = 'E'.toByte()
        header[12] = 'f'.toByte()
        header[13] = 'm'.toByte()
        header[14] = 't'.toByte()
        header[15] = ' '.toByte()
        header[16] = 16 // Header size
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // Format: PCM
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = (sampleRate shr 8 and 0xff).toByte()
        header[26] = (sampleRate shr 16 and 0xff).toByte()
        header[27] = (sampleRate shr 24 and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = (byteRate shr 8 and 0xff).toByte()
        header[30] = (byteRate shr 16 and 0xff).toByte()
        header[31] = (byteRate shr 24 and 0xff).toByte()
        header[32] = (channels * 16 / 8).toByte() // Block align
        header[33] = 0
        header[34] = 16 // Bits per sample
        header[35] = 0
        header[36] = 'd'.toByte()
        header[37] = 'a'.toByte()
        header[38] = 't'.toByte()
        header[39] = 'a'.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = (totalAudioLen shr 8 and 0xff).toByte()
        header[42] = (totalAudioLen shr 16 and 0xff).toByte()
        header[43] = (totalAudioLen shr 24 and 0xff).toByte()

        RandomAccessFile(file, "rw").use { raf ->
            raf.seek(0)
            raf.write(header)
        }
    }
}
