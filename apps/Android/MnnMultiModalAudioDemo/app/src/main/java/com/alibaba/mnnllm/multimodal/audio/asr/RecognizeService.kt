package com.alibaba.mnnllm.multimodal.audio.asr

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import com.alibaba.mnnllm.multimodal.audio.MainActivity
import com.k2fsa.sherpa.mnn.OnlineCtcFstDecoderConfig
import com.k2fsa.sherpa.mnn.OnlineRecognizer
import com.k2fsa.sherpa.mnn.OnlineRecognizerConfig
import com.k2fsa.sherpa.mnn.getEndpointConfig
import com.k2fsa.sherpa.mnn.getFeatureConfig
import com.k2fsa.sherpa.mnn.OnlineLMConfig
import com.k2fsa.sherpa.mnn.OnlineModelConfig
import com.k2fsa.sherpa.mnn.OnlineTransducerModelConfig
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class RecognizeService(private val activity: MainActivity) {
    companion object {
        const val TAG = "ASR_RecognizeService"
        private const val REQUEST_RECORD_AUDIO_PERMISSION = 1001
    }

    private val initComplete = CompletableDeferred<Boolean>()
    private var recognizer: OnlineRecognizer? = null
    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null

    private val sampleRateInHz = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val isRecording = AtomicBoolean(false)

    @Volatile
    private var isLoaded = false
    var onRecognizeText: ((String) -> Unit)? = null

    private val acceptTimeNs = AtomicLong(0)
    private val decodeTimeNs = AtomicLong(0)
    private val chunkCount = AtomicLong(0)
    private val decodeCount = AtomicLong(0)

    private var utteranceProcTimeNs: Long = 0
    private var utteranceAudioTimeSec: Double = 0.0
    private var totalRtf: Double = 0.0
    private var utteranceCount: Int = 0

    suspend fun initRecognizer(asrModelDir: String, int8: Boolean = true) {
        val config = OnlineRecognizerConfig(
            featConfig = getFeatureConfig(sampleRateInHz, 80),
            modelConfig = OnlineModelConfig(
                transducer = if (int8) {
                    OnlineTransducerModelConfig(
                        encoder = "$asrModelDir/encoder-epoch-99-avg-1.int8.mnn",
                        decoder = "$asrModelDir/decoder-epoch-99-avg-1.int8.mnn",
                        joiner = "$asrModelDir/joiner-epoch-99-avg-1.int8.mnn",
                    )
                } else {
                    OnlineTransducerModelConfig(
                        encoder = "$asrModelDir/encoder-epoch-99-avg-1.mnn",
                        decoder = "$asrModelDir/decoder-epoch-99-avg-1.mnn",
                        joiner = "$asrModelDir/joiner-epoch-99-avg-1.mnn",
                    )
                },
                tokens = "$asrModelDir/tokens.txt",
                modelType = "zipformer",
            ),
            lmConfig = OnlineLMConfig(
                model = "$asrModelDir/with-state-epoch-99-avg-1.int8.onnx",
                scale = 0.5f
            ),
            ctcFstDecoderConfig = OnlineCtcFstDecoderConfig("", 3000),
            endpointConfig = getEndpointConfig(),
            enableEndpoint = true,
            decodingMethod = "greedy_search",
            maxActivePaths = 4,
            hotwordsFile = "",
            hotwordsScore = 1.5f,
            ruleFsts = "",
            ruleFars = "",
            blankPenalty = 0.0f,
        )
        CoroutineScope(Dispatchers.IO).async {
            recognizer = OnlineRecognizer(null, config)
        }.await()
        isLoaded = true
        initComplete.complete(true)
    }

    fun isRecording(): Boolean = isRecording.get()

    private fun initMicrophone(): Boolean {
        if (ActivityCompat.checkSelfPermission(
                activity,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(activity, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO_PERMISSION)
            return false
        }
        val minBufferBytes = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat)
        if (minBufferBytes <= 0) {
            Log.e(TAG, "Invalid min buffer size: $minBufferBytes")
            return false
        }
        val bufferBytes = minBufferBytes * 2
        audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRateInHz, channelConfig, audioFormat, bufferBytes)
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized")
            audioRecord?.release()
            audioRecord = null
            return false
        }
        return true
    }

    private fun processSamples() {
        val localRecognizer = recognizer
        val localAudioRecord = audioRecord
        if (localRecognizer == null || localAudioRecord == null || !isLoaded) {
            isRecording.set(false)
            return
        }
        val stream = localRecognizer.createStream("")
        val interval = 0.1
        val targetSamples = (interval * sampleRateInHz).toInt()
        val minBufferBytes = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat)
        val minSamples = if (minBufferBytes > 0) minBufferBytes / 2 else targetSamples
        val bufferSize = maxOf(targetSamples, minSamples)
        val buffer = ShortArray(bufferSize)

        try {
            while (isRecording.get() && audioRecord != null) {
                val ret = localAudioRecord.read(buffer, 0, buffer.size)
                if (ret == AudioRecord.ERROR_INVALID_OPERATION || ret == AudioRecord.ERROR_BAD_VALUE) {
                    Log.e(TAG, "AudioRecord read error: $ret")
                    break
                }
                if (ret <= 0) {
                    continue
                }
                chunkCount.incrementAndGet()
                val samples = FloatArray(ret) { i -> buffer[i] / 32768.0f }
                utteranceAudioTimeSec += ret.toDouble() / sampleRateInHz
                val tStartProc = System.nanoTime()
                val tAcceptStart = tStartProc
                stream.acceptWaveform(samples, sampleRateInHz)
                val tAcceptEnd = System.nanoTime()
                acceptTimeNs.addAndGet(tAcceptEnd - tAcceptStart)
                while (localRecognizer.isReady(stream)) {
                    val tDecodeStart = System.nanoTime()
                    localRecognizer.decode(stream)
                    val tDecodeEnd = System.nanoTime()
                    decodeTimeNs.addAndGet(tDecodeEnd - tDecodeStart)
                    decodeCount.incrementAndGet()
                }
                val tEndProc = System.nanoTime()
                utteranceProcTimeNs += (tEndProc - tStartProc)

                val isEndpoint = localRecognizer.isEndpoint(stream)
                val text = localRecognizer.getResult(stream).text

                if (isEndpoint) {
                    val T_proc = utteranceProcTimeNs / 1_000_000_000.0
                    val T_audio = utteranceAudioTimeSec
                    val rtf = if (T_audio > 0) T_proc / T_audio else 0.0
                    totalRtf += rtf
                    utteranceCount += 1
                    localRecognizer.reset(stream)
                    if (text.isNotEmpty()) {
                        onRecognizeText?.invoke(text)
                        Log.d(TAG, "recognize text: $text")
                    }
                    utteranceProcTimeNs = 0
                    utteranceAudioTimeSec = 0.0
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "processSamples failed", e)
        } finally {
            stream.release()
            acceptTimeNs.set(0)
            decodeTimeNs.set(0)
            chunkCount.set(0)
            decodeCount.set(0)
            totalRtf = 0.0
            utteranceCount = 0
            stopRecord()
        }
    }

    fun stopRecord() {
        if (!isRecording.get()) {
            return
        }
        isRecording.set(false)
        val thread = recordingThread
        audioRecord?.let {
            it.stop()
            it.release()
            audioRecord = null
        }
        recordingThread = null
        if (thread != null && thread != Thread.currentThread()) {
            try {
                thread.join(500)
            } catch (e: InterruptedException) {
                Log.e(TAG, "Join recording thread interrupted", e)
            }
        }
    }

    fun startRecord() {
        if (isRecording.get()) {
            return
        }
        if (!initComplete.isCompleted || !isLoaded || recognizer == null) {
            Log.e(TAG, "ASR not ready")
            return
        }
        val ret = initMicrophone()
        if (!ret) {
            Log.e(TAG, "Failed to initialize microphone")
            return
        }
        audioRecord?.let {
            it.startRecording()
            isRecording.set(true)
            val thread = Thread { this.processSamples() }
            thread.name = "AsrRecorder"
            recordingThread = thread
            thread.start()
        }
    }
}
