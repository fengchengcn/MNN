package com.alibaba.mnnllm.multimodal.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import kotlin.math.abs
import kotlin.math.min
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
    private var outputFile: File? = null

    var onAmplitudeUpdate: ((Int) -> Unit)? = null

    @SuppressLint("MissingPermission")
    fun startRecording(outputFilePath: String) {
        if (isRecording) return

        val file = File(outputFilePath)
        outputFile = file
        val audioSource =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                    MediaRecorder.AudioSource.UNPROCESSED
                } else {
                    MediaRecorder.AudioSource.MIC
                }
        audioRecord =
                AudioRecord(
                        audioSource,
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

    suspend fun stopRecording(): File? {
        isRecording = false

        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recorder: ${e.message}")
        }
        audioRecord = null
        val job = recordingJob
        recordingJob = null
        job?.join()
        val file = outputFile
        outputFile = null
        return file
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
                        val finalShort = shortBuffer[i]
                        val abs = Math.abs(finalShort.toInt())
                        if (abs > maxAmplitude) maxAmplitude = abs
                        byteBuffer.putShort(finalShort)
                    }
                    onAmplitudeUpdate?.invoke(maxAmplitude)
                    out.write(byteBuffer.array(), 0, read * 2)
                }
            }
        }
        updateWavHeader(file)
        normalizeWavPcm(file)
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

    private fun normalizeWavPcm(file: File) {
        if (file.length() <= 44) return
        RandomAccessFile(file, "rw").use { raf ->
            val dataSize = (raf.length() - 44).toInt()
            if (dataSize <= 0) return
            val bytes = ByteArray(dataSize)
            raf.seek(44)
            raf.readFully(bytes)
            var peak = 1
            var i = 0
            while (i < bytes.size - 1) {
                val sample =
                        (bytes[i].toInt() and 0xff) or
                                (bytes[i + 1].toInt() shl 8)
                val s = sample.toShort().toInt()
                val a = abs(s)
                if (a > peak) peak = a
                i += 2
            }
            val target = (0.8f * 32767f).toInt()
            val gain = target.toFloat() / peak.toFloat()
            val applyGain = min(gain, 2.5f)
            if (applyGain <= 1.05f) return
            i = 0
            while (i < bytes.size - 1) {
                val sample =
                        (bytes[i].toInt() and 0xff) or
                                (bytes[i + 1].toInt() shl 8)
                val s = sample.toShort().toInt()
                val scaled =
                        (s.toFloat() * applyGain)
                                .coerceIn(Short.MIN_VALUE.toFloat(), Short.MAX_VALUE.toFloat())
                                .toInt()
                bytes[i] = (scaled and 0xff).toByte()
                bytes[i + 1] = (scaled shr 8 and 0xff).toByte()
                i += 2
            }
            raf.seek(44)
            raf.write(bytes)
        }
    }
}
