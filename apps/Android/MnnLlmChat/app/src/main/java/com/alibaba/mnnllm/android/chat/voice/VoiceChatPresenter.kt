// Created by ruoyi.sjd on 2025/06/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat.voice

import android.app.Activity
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import android.util.Log
import com.alibaba.mnnllm.android.asr.AsrService
import com.alibaba.mnnllm.android.asr.Qwen3AsrEngine
import com.alibaba.mnnllm.android.audio.AudioChunksPlayer
import com.alibaba.mnnllm.android.chat.ChatPresenter
import com.alibaba.mnnllm.android.chat.GenerateResultProcessor
import com.alibaba.mnnllm.android.utils.VoiceModelPathUtils
import com.taobao.meta.avatar.tts.TtsService
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.consumeEach
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class VoiceChatPresenterState {
    INITIALIZING,
    LISTENING,
    GENERATING_TEXT,
    PLAYING,
    PLAY_END
}

// Sealed class for sequential tasks
sealed class SerialTask {
    data class ProcessProgress(val progress: String, val isFirstChunk: Boolean, val responseBuilder: StringBuilder, val ttsSegmentBuffer: StringBuilder) : SerialTask()
    data class ProcessFinalChunk(val ttsSegmentBuffer: StringBuilder) : SerialTask()
    data class HandleAsrResult(val text: String) : SerialTask()
    object OnTtsComplete : SerialTask()
}

enum class VoiceChatState {
    CONNECTING,
    GREETING,
    LISTENING,
    PROCESSING,
    ASR_DECODING,   // Qwen3-ASR streaming decode in progress (partial result updates)
    THINKING,
    SPEAKING,
    STOPPING,
    ERROR
}

/**
 * ASR engine mode:
 * - SHERPA: Original Sherpa MNN ASR (separate ASR engine, text-only LLM)
 * - QWEN3_OLD: Old Qwen3-ASR engine (qwen3_asr_engine.cpp, CPU only, manual decode loop)
 * - QWEN3_OMNI: New Omni engine path (LlmSession via <audio> tag, supports GPU)
 */
private enum class AsrMode {
    SHERPA,
    QWEN3_OLD,
    QWEN3_OMNI
}

class VoiceChatPresenter(
    private val activity: Activity,
    private val view: VoiceChatView,
    private val chatPresenter: ChatPresenter,
    private val lifecycleScope: CoroutineScope,
    private val ttsClientFactory: () -> TtsClient = { RealTtsClient(TtsService()) }
) : ChatPresenter.GenerateListener {
    companion object {
        const val TAG = "VoiceChatPresenter"
    }

    // --- Sherpa MNN ASR ---
    private var asrService: AsrService? = null

    // --- Qwen3-ASR ---
    private var qwen3AsrEngine: Qwen3AsrEngine? = null
    private var qwen3AudioRecord: AudioRecord? = null
    private var qwen3Aec: AcousticEchoCanceler? = null
    private var qwen3Ns: NoiseSuppressor? = null
    private var qwen3RecordingThread: Thread? = null
    private var asrMode = AsrMode.SHERPA
    private val qwen3IsRecording = AtomicBoolean(false)
    // Silence-based endpoint detection for Qwen3-ASR
    private var silenceChunkCount = 0
    private var speechDetectedForQwen3 = false
    private val silenceRmsThreshold = 100.0f   // RMS on raw int16 PCM below this = silence
    private val speechRmsThreshold = 400.0f    // RMS on raw int16 PCM above this = speech (for interruption)
    private val maxSilenceChunks = 15           // ~1.5s silence triggers endpoint
    private val qwen3SampleRate = 16000
    // Omni audio mode: collect PCM samples, write WAV, send <audio> tag
    private val omniAudioBuffer = mutableListOf<Float>()
    private val omniWavDir: String by lazy {
        File(activity.cacheDir, "omni_audio").also { it.mkdirs() }.absolutePath
    }

    private var ttsService: TtsClient? = null
    private var audioPlayer: AudioChunksPlayer? = null
    private var audioManager: AudioManager = activity.getSystemService(Activity.AUDIO_SERVICE) as AudioManager

    private var isRecording = false
    private var isSpeaking = false
    private var isProcessingLlm = false
    private var isStopped = false
    private var isStoppingGeneration = false
    private var isGenerationFinished = false
    private var isMuted = false
    private var isAutoMuteForEchoCancelMode = false
    
    // For handling LLM generation progress with thinking support
    private var generateResultProcessor: GenerateResultProcessor? = null
    private var responseBuilder = StringBuilder()
    private var ttsSegmentBuffer = StringBuilder()
    private var isFirstChunk = true
    private var isThinking = false
    
    private var currentStatus: VoiceChatPresenterState = VoiceChatPresenterState.INITIALIZING
        set(value) {
            if (field != value) {
                Log.d(TAG, "Status changed from ${field.name} to ${value.name}")
                field = value
            }
        }

    // Channel-based sequential processor
    private val taskChannel = Channel<SerialTask>(Channel.UNLIMITED)
    private val serialProcessor = lifecycleScope.launch {
        taskChannel.consumeEach { task ->
            if (!isStopped) {
                processTask(task)
            }
        }
    }

    private suspend fun processTask(task: SerialTask) {
        when (task) {
            is SerialTask.ProcessProgress -> {
                if (isStoppingGeneration) return
                Log.d(TAG, "progress is ${task.progress}")
                
                if (task.isFirstChunk) {
                    // Initialize processor for new generation
                    generateResultProcessor = GenerateResultProcessor()
                    generateResultProcessor?.generateBegin()
                    withContext(Dispatchers.Main) { view.addTranscript(Transcript(isUser = false, text = "")) }
                }
                
                // Process the progress through GenerateResultProcessor
                generateResultProcessor?.process(task.progress)
                
                // Check if we're in thinking mode
                val thinkingContent = generateResultProcessor?.getThinkingContent() ?: ""
                val normalOutput = generateResultProcessor?.getNormalOutput() ?: ""
                val wasThinking = isThinking
                isThinking = thinkingContent.isNotEmpty() && normalOutput.isEmpty()
                
                // Update status based on thinking state
                if (isThinking && !wasThinking) {
                    // Just entered thinking mode
                    withContext(Dispatchers.Main) { view.updateStatus(VoiceChatState.THINKING) }
                    Log.d(TAG, "Entering thinking mode")
                } else if (!isThinking && wasThinking) {
                    // Just exited thinking mode
                    withContext(Dispatchers.Main) { view.updateStatus(VoiceChatState.PROCESSING) }
                    Log.d(TAG, "Exiting thinking mode")
                }
                
                // Only show normal output in transcripts (not thinking content)
                if (normalOutput.isNotEmpty()) {
                    Log.d(TAG, "Normal output is not empty: '$normalOutput' progress: ${task.progress}")
                    task.responseBuilder.clear()
                    task.responseBuilder.append(normalOutput)
                    withContext(Dispatchers.Main) { view.updateLastTranscript(normalOutput) }
                    
                    // Process TTS for normal output only
                    val delimiters = "[.,!。，！？?\n、：；:]".toRegex()
                    val progressText = GenerateResultProcessor.noSlashThink(task.progress)!!
                    task.ttsSegmentBuffer.append(progressText)
                    if (delimiters.containsMatchIn(progressText) && !isThinking) {
                        val textToSpeak = task.ttsSegmentBuffer.toString()
                        task.ttsSegmentBuffer.clear()
                        Log.d(TAG, "Delimiter found. Speaking: '$textToSpeak'")
                        if (!isStopped && !isStoppingGeneration) {
                            currentStatus = VoiceChatPresenterState.PLAYING
                            withContext(Dispatchers.Main) { view.updateStatus(VoiceChatState.SPEAKING) }
                            val audioData = processTtsText(textToSpeak)
                            if (audioData != null && audioData.isNotEmpty() && !isStopped && !isStoppingGeneration) {
                                audioPlayer?.playChunk(audioData)
                            }
                        }
                    }
                }
                
                Log.d(TAG, "progress is ${task.progress} end")
            }
            is SerialTask.ProcessFinalChunk -> {
                if (isStoppingGeneration) return
                Log.d(TAG, "progress is null")
                
                // Process final chunk through GenerateResultProcessor
                generateResultProcessor?.process(null)
                
                // Reset thinking state
                isThinking = false
                
                if (task.ttsSegmentBuffer.isNotEmpty()) {
                    val textToSpeak = task.ttsSegmentBuffer.toString()
                    task.ttsSegmentBuffer.clear()
                    Log.d(TAG, "Speaking remaining buffer: '$textToSpeak'")
                    currentStatus = VoiceChatPresenterState.PLAYING
                    withContext(Dispatchers.Main) { view.updateStatus(VoiceChatState.SPEAKING) }
                    val audioData = processTtsText(textToSpeak)
                    if (audioData != null && audioData.isNotEmpty() && !isStopped && !isStoppingGeneration) {
                        audioPlayer?.playChunk(audioData)
                    }
                }
                if (!isStoppingGeneration) {
                    audioPlayer?.endChunk()
                }
                Log.d(TAG, "progress is null end")
            }
            is SerialTask.HandleAsrResult -> {
                if (isStoppingGeneration) return
                isProcessingLlm = true
                isSpeaking = true
                isThinking = false
                currentStatus = VoiceChatPresenterState.GENERATING_TEXT
                withContext(Dispatchers.Main) {
                    // For Omni mode, the "text" is an <audio> tag — show a friendly label
                    val displayText = if (asrMode == AsrMode.QWEN3_OMNI) {
                        activity.getString(com.alibaba.mnnllm.android.R.string.voice_chat_audio_input)
                    } else {
                        task.text
                    }
                    view.addTranscript(Transcript(isUser = true, text = displayText))
                    view.updateStatus(VoiceChatState.PROCESSING)
                }
                // Automatically mute microphone in Auto-Mute mode when AI starts processing/speaking
                if (isAutoMuteForEchoCancelMode) {
                    muteMicrophone(true)
                }
                // We don't call `stopRecord()` here to keep ASR active during LLM generation to support "speech interruption" (full-duplex). If the user starts speaking, onSpeechDetected will trigger and stop current generation.
                // stopRecord()

                // Check if a vision-mode photo has been captured and is ready for sending
                val capturedImageUri = view.getCapturedImageUri()
                if (capturedImageUri != null) {
                    // --- Vision Mode Execution Path ---
                    // If an image is present, we trigger a multi-modal interaction.
                    // This allows the AI to "see" what the camera is currently looking at.
                    Log.i(TAG, "Vision Mode: Processing message with captured image: $capturedImageUri")
                    
                    // Construct a ChatDataItem compatible with ChatPresenter's multi-modal message format
                    val userData = com.alibaba.mnnllm.android.chat.model.ChatDataItem(com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders.USER)
                    userData.text = task.text
                    userData.imageUris = listOf(capturedImageUri) // Attach the captured photo
                    userData.time = chatPresenter.dateFormat.format(java.util.Date())

                    // Reset local generation/playback states to prepare for a fresh response
                    responseBuilder.clear()
                    ttsSegmentBuffer.clear()
                    isFirstChunk = true
                    isGenerationFinished = false

                    // Delegate the actual message sending and LLM interaction to the main ChatPresenter
                    lifecycleScope.launch(Dispatchers.IO) {
                        chatPresenter.sendMessage(userData)
                    }
                    
                    // Crucial: Clear the captured image URI to ensure it doesn't persist to the next turn erroneously
                    view.clearCapturedImageUri()
                } else {
                    // --- Standard Voice Mode Execution Path ---
                    // No image present; perform standard text-based LLM generation
                    Log.d(TAG, "Standard Mode: Sending text-only generation request: ${task.text}")
                    llmGenerate(task.text)
                }
            }
            is SerialTask.OnTtsComplete -> {
                // Always handle TTS completion to ensure proper state transition
                Log.d(TAG, "TTS playback completed, transitioning to LISTENING state")
                isProcessingLlm = false
                isSpeaking = false
                isThinking = false
                currentStatus = VoiceChatPresenterState.LISTENING
                withContext(Dispatchers.Main) {
                    view.updateStatus(VoiceChatState.LISTENING)
                }
                audioPlayer?.reset()
                kotlinx.coroutines.delay(500)
                // Only start recording if we're not in the middle of stopping
                if (!isStoppingGeneration) {
                    // Automatically un-mute microphone in Auto-Mute mode when AI finishes speaking
                    if (isAutoMuteForEchoCancelMode) {
                        muteMicrophone(false)
                    }
                startRecord()
                }
            }
        }
    }

    fun start() {
        Log.d(TAG, "Presenter starting...")
        isStopped = false
        isGenerationFinished = false
        currentStatus = VoiceChatPresenterState.INITIALIZING
        
        // Register this presenter as an additional listener to ChatPresenter
        chatPresenter.addGenerateListener(this)

        view.updateMuteButtonState(isMuted)
        view.updateEchoCancelMode(isAutoMuteForEchoCancelMode)
        
        initTts()
        startAsr()
    }


    private fun initAudio(sampleRate: Int) {
        // Clean up existing audio player first
        audioPlayer?.destroy()
        
        audioPlayer = AudioChunksPlayer()
        
        // Set up the completion listener with more detailed logging
        audioPlayer?.setOnCompletionListener {
            Log.d(TAG, "Audio playback completed - currentStatus: ${currentStatus.name}, isSpeaking: $isSpeaking, isProcessingLlm: $isProcessingLlm")
            currentStatus = VoiceChatPresenterState.PLAY_END
            lifecycleScope.launch {
                Log.d(TAG, "Sending OnTtsComplete task")
                taskChannel.send(SerialTask.OnTtsComplete)
            }
        }
        
        audioPlayer?.sampleRate = sampleRate
        audioPlayer?.start()
        Log.d(TAG, "Audio player initialized with completion listener, sampleRate=$sampleRate")
    }

    private fun initTts() {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                if (isStopped) return@launch
                
                Log.d(TAG, "Initializing TTS Service...")
                ttsService = ttsClientFactory()
                val modelDir = VoiceModelPathUtils.getTtsModelPath(activity)
                val sampleRate = VoiceModelPathUtils.getTtsSampleRate(modelDir)
                val language = VoiceModelPathUtils.getTtsLanguage(activity)
                ttsService?.setLanguage(language)
                initAudio(sampleRate)
                withContext(Dispatchers.IO) {
                    if (isStopped) return@withContext
                    
                    Log.i(TAG, "Using TTS model path: $modelDir")
                    Log.i(TAG, "Using TTS language: $language")
                    val initResult = ttsService?.init(modelDir)
                    if (initResult != true) {
                        Log.e(TAG, "TTS Service initialization failed with path: $modelDir")
                        if (!isStopped) withContext(Dispatchers.Main) { view.showError("TTS init failed") }
                    } else {
                        Log.d(TAG, "TTS Service initialized successfully with path: $modelDir")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "TTS initialization failed", e)
                if (!isStopped) withContext(Dispatchers.Main) { view.showError("TTS init failed: ${e.message}") }
            }
        }
    }

    /**
     * Detect which ASR engine mode to use for the given model directory.
     * - QWEN3_OMNI: audio.mnn + config.json with is_audio=true (new llmexport.py path)
     * - QWEN3_OLD: audio_encoder.mnn exists (legacy export path)
     * - SHERPA: default
     */
    private fun detectAsrMode(modelDir: String): AsrMode {
        val dir = File(modelDir)
        // New Omni path: audio.mnn + config.json is_audio=true
        if (File(dir, "audio.mnn").exists() && File(dir, "config.json").exists()) {
            try {
                val config = JSONObject(File(dir, "config.json").readText())
                if (config.optBoolean("is_audio", false)) {
                    Log.i(TAG, "Omni audio model detected: $modelDir")
                    return AsrMode.QWEN3_OMNI
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to parse config.json: ${e.message}")
            }
        }
        // Old path: audio_encoder.mnn
        if (File(dir, "audio_encoder.mnn").exists()) {
            Log.i(TAG, "Legacy Qwen3-ASR model detected: $modelDir")
            return AsrMode.QWEN3_OLD
        }
        Log.d(TAG, "Sherpa ASR mode (no audio model files found): $modelDir")
        return AsrMode.SHERPA
    }

    private fun startAsr() {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                if (isStopped) return@launch

                Log.d(TAG, "Initializing ASR Service...")
                val modelDir = VoiceModelPathUtils.getAsrModelPath(activity)
                Log.i(TAG, "Using ASR model path: $modelDir")

                // --- Detect ASR engine mode ---
                asrMode = detectAsrMode(modelDir)

                when (asrMode) {
                    AsrMode.QWEN3_OMNI -> startQwen3AsrOmni(modelDir)
                    AsrMode.QWEN3_OLD -> startQwen3Asr(modelDir)
                    AsrMode.SHERPA -> startSherpaAsr(modelDir)
                }
            } catch (e: Exception) {
                Log.e(TAG, "ASR initialization or start failed", e)
                if (!isStopped) withContext(Dispatchers.Main) { view.showError("ASR init failed: ${e.message}") }
            }
        }
    }

    // ==================== Sherpa MNN ASR (original flow) ====================

    private suspend fun startSherpaAsr(modelDir: String) {
        asrService = AsrService(activity, modelDir)

        withContext(Dispatchers.IO) {
            if (isStopped) return@withContext
            asrService?.initRecognizer()
        }

        if (isStopped) return

        asrService?.onRecognizeText = { text ->
            lifecycleScope.launch {
                if (!isStopped && text.isNotEmpty() && !isSpeaking && !isProcessingLlm) {
                    Log.i(TAG, "ASR Result: $text")
                    taskChannel.send(SerialTask.HandleAsrResult(text))
                } else {
                    Log.d(TAG, "ASR ignored: text='$text', isSpeaking=$isSpeaking, isProcessingLlm=$isProcessingLlm, isStopped=$isStopped")
                }
            }
        }

        asrService?.onSpeechDetected = {
            lifecycleScope.launch(Dispatchers.Main) {
                if (!isStopped && (isSpeaking || isProcessingLlm)) {
                    Log.i(TAG, "Speech detected during AI output, interrupting...")
                    stopGeneration()
                }
                if (view.isCameraEnabled() && !isSpeaking && !isProcessingLlm) {
                    Log.d(TAG, "Speech detected, capturing photo...")
                    view.capturePhoto()
                }
            }
        }

        isGenerationFinished = false
        startRecord()
        currentStatus = VoiceChatPresenterState.LISTENING
        if (!isStopped) withContext(Dispatchers.Main) {
            view.updateStatus(VoiceChatState.LISTENING)
            view.showGreetingMessage()
            speakGreetingMessage()
        }
        Log.i(TAG, "Sherpa ASR started. Now listening.")
    }

    // ==================== Qwen3-ASR (batch processing) ====================

    private suspend fun startQwen3Asr(modelDir: String) {
        withContext(Dispatchers.IO) {
            if (isStopped) return@withContext

            qwen3AsrEngine = Qwen3AsrEngine()
            val ok = qwen3AsrEngine!!.init(modelDir, activity.cacheDir.absolutePath, numThreads = 4)
            if (!ok) {
                Log.e(TAG, "Qwen3AsrEngine init failed")
                withContext(Dispatchers.Main) { view.showError("Qwen3-ASR init failed") }
                return@withContext
            }
            Log.i(TAG, "Qwen3AsrEngine initialized successfully")
        }

        if (isStopped) return

        isGenerationFinished = false
        startQwen3Record()
        currentStatus = VoiceChatPresenterState.LISTENING
        if (!isStopped) withContext(Dispatchers.Main) {
            view.updateStatus(VoiceChatState.LISTENING)
            view.showGreetingMessage()
            speakGreetingMessage()
        }
        Log.i(TAG, "Qwen3-ASR started. Now listening.")
    }

    // ==================== Omni Audio (new LlmSession path, supports GPU) ====================

    /**
     * Start Omni audio mode. No separate ASR engine loaded — audio is recorded,
     * written as WAV, and sent to the Omni engine via <audio> tag through ChatPresenter.
     * The Omni engine (LlmSession) handles: fbank → AE → embedding injection → inference.
     */
    private suspend fun startQwen3AsrOmni(modelDir: String) {
        Log.i(TAG, "Starting Omni audio mode (no separate ASR engine)")
        // No Qwen3AsrEngine loading — Omni engine is already loaded via LlmSession
        // from ChatPresenter's session initialization.

        // Clean up WAV files older than 1 hour to prevent accumulation
        try {
            val staleThreshold = System.currentTimeMillis() - 3600000L
            File(omniWavDir).listFiles()?.forEach { f ->
                if (f.name.startsWith("omni_") && f.lastModified() < staleThreshold) {
                    f.delete()
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Omni WAV cleanup failed: ${e.message}")
        }

        if (isStopped) return

        isGenerationFinished = false
        startQwen3Record()
        currentStatus = VoiceChatPresenterState.LISTENING
        if (!isStopped) withContext(Dispatchers.Main) {
            view.updateStatus(VoiceChatState.LISTENING)
            view.showGreetingMessage()
            speakGreetingMessage()
        }
        Log.i(TAG, "Omni audio started. Now listening.")
    }

    private fun startQwen3Record() {
        if (qwen3IsRecording.get()) return

        val minBufSize = AudioRecord.getMinBufferSize(qwen3SampleRate,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        qwen3AudioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            qwen3SampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBufSize * 2
        )

        // Try to enable AEC and NS (same as AsrService)
        try {
            if (AcousticEchoCanceler.isAvailable()) {
                qwen3Aec = AcousticEchoCanceler.create(qwen3AudioRecord!!.audioSessionId)
                qwen3Aec?.enabled = true
                Log.i(TAG, "Qwen3: AEC enabled")
            }
        } catch (_: Exception) {}
        try {
            if (NoiseSuppressor.isAvailable()) {
                qwen3Ns = NoiseSuppressor.create(qwen3AudioRecord!!.audioSessionId)
                qwen3Ns?.enabled = true
                Log.i(TAG, "Qwen3: NS enabled")
            }
        } catch (_: Exception) {}

        qwen3AudioRecord!!.startRecording()
        qwen3IsRecording.set(true)
        isRecording = true
        silenceChunkCount = 0
        speechDetectedForQwen3 = false

        qwen3RecordingThread = Thread { processQwen3Samples() }
        qwen3RecordingThread!!.start()
        Log.i(TAG, "Qwen3 recording started")
    }

    private fun processQwen3Samples() {
        val interval = 0.1  // 100ms chunks
        val chunkSize = (interval * qwen3SampleRate).toInt()
        val shortBuf = ShortArray(chunkSize)
        val engine = qwen3AsrEngine  // Non-null only in QWEN3_OLD mode — guarded by asrMode checks

        val maxChunks = 300  // 30s max recording to prevent infinite loop
        var totalChunks = 0

        Log.i(TAG, "Qwen3 sample processing started")

        while (qwen3IsRecording.get() && qwen3AudioRecord != null && totalChunks < maxChunks) {
            totalChunks++
            val ret = qwen3AudioRecord!!.read(shortBuf, 0, chunkSize)
            if (ret <= 0) continue

            // Mute handling
            if (isMuted) {
                shortBuf.fill(0)
            }

            // Convert int16 → float32 [-1, 1] (for engine input)
            val floatBuf = FloatArray(ret) { i -> shortBuf[i] / 32768.0f }

            // RMS energy for silence/speech detection (on raw int16 PCM, same as Qwen3AsrTestActivity)
            var sumSq = 0.0f
            for (s in shortBuf) sumSq += (s.toFloat() * s.toFloat())
            val rms = kotlin.math.sqrt(sumSq / ret)

            if (rms > speechRmsThreshold) {
                speechDetectedForQwen3 = true
                // Interruption support: if AI is speaking and user starts talking
                if (!isStopped && (isSpeaking || isProcessingLlm)) {
                    Log.i(TAG, "Qwen3 speech detected during AI output, interrupting...")
                    lifecycleScope.launch(Dispatchers.Main) { stopGeneration() }
                }
            }

            if (speechDetectedForQwen3 && rms < silenceRmsThreshold) {
                silenceChunkCount++
            } else if (rms >= silenceRmsThreshold) {
                silenceChunkCount = 0
            }

            // Push audio to engine (even during silence — engine needs full utterance)
            if (asrMode == AsrMode.QWEN3_OMNI) {
                // Omni mode: collect audio in buffer, will write WAV at endpoint
                synchronized(omniAudioBuffer) {
                    omniAudioBuffer.addAll(floatBuf.asList())
                }
            } else {
                engine.pushAudio(floatBuf)
            }

            // Endpoint: sustained silence after speech
            if (speechDetectedForQwen3 && silenceChunkCount >= maxSilenceChunks) {
                Log.i(TAG, "Qwen3 endpoint detected (${silenceChunkCount} silence chunks)")
                break
            }
        }

        if (totalChunks >= maxChunks) {
            Log.w(TAG, "Qwen3 max recording duration reached, forcing endpoint")
            speechDetectedForQwen3 = true  // force decode
        }

        // Stop recording
        qwen3IsRecording.set(false)
        qwen3AudioRecord?.stop()
        qwen3AudioRecord?.release()
        qwen3AudioRecord = null
        isRecording = false

        if (speechDetectedForQwen3) {
            if (asrMode == AsrMode.QWEN3_OMNI) {
                // ====== Omni mode: write WAV and send <audio> tag through LlmSession ======
                // Bail early if presenter was stopped while recording (avoids empty WAV I/O)
                if (isStopped) {
                    omniAudioBuffer.clear()
                    return
                }
                if (omniAudioBuffer.isEmpty()) {
                    Log.w(TAG, "Omni: no audio data collected, restarting")
                    kotlinx.coroutines.delay(200)
                    if (!isStopped) startQwen3Record()
                    return
                }
                lifecycleScope.launch {
                    Log.i(TAG, "Omni: writing WAV from ${omniAudioBuffer.size} samples...")
                    val wavFileName = "omni_${UUID.randomUUID()}.wav"
                    val wavFile = File(omniWavDir, wavFileName)
                    val ok = withContext(Dispatchers.IO) {
                        writeWavFile(omniAudioBuffer.toFloatArray(), qwen3SampleRate, wavFile.absolutePath)
                    }
                    omniAudioBuffer.clear()
                    if (ok && !isStopped && !isSpeaking && !isProcessingLlm) {
                        val audioTag = "<audio>${wavFile.absolutePath}</audio>"
                        Log.i(TAG, "Omni: sending audio prompt: $audioTag")
                        taskChannel.send(SerialTask.HandleAsrResult(audioTag))
                    } else if (!isStopped) {
                        Log.w(TAG, "Omni: WAV write failed or busy, restarting")
                        kotlinx.coroutines.delay(200)
                        if (!isStopped) startQwen3Record()
                    }
                }
            } else {
                // ====== Old Qwen3-ASR: run streaming decode ======
                Log.i(TAG, "Qwen3 starting streaming decode...")
                lifecycleScope.launch {
                    withContext(Dispatchers.Main) {
                        view.updateStatus(VoiceChatState.ASR_DECODING)
                    }
                }

                val ok = engine.startDecode()
                Log.i(TAG, "Qwen3 startDecode: $ok, isDecoding=${engine.isDecoding()}")

                if (ok) {
                    var lastPartial = ""
                    while (engine.isDecoding() && !isStopped) {
                        engine.decodeStep()
                        val partialText = engine.getPartialResult()
                        if (partialText.isNotEmpty() && partialText != lastPartial) {
                            lastPartial = partialText
                            Log.d(TAG, "Qwen3 ASR partial: $partialText")
                            lifecycleScope.launch {
                                withContext(Dispatchers.Main) {
                                    view.updateAsrPartialText(partialText)
                                }
                            }
                        }
                    }
                    val text = engine.getResultText()
                    Log.i(TAG, "Qwen3 ASR final: $text, tokens=${engine.getResult()}")
                    engine.reset()

                    if (text.isNotEmpty() && !isStopped && !isSpeaking && !isProcessingLlm) {
                        lifecycleScope.launch {
                            taskChannel.send(SerialTask.HandleAsrResult(text))
                        }
                    } else if (!isStopped) {
                        // No text produced — restart listening
                        lifecycleScope.launch {
                            kotlinx.coroutines.delay(200)
                            if (!isStopped) startQwen3Record()
                        }
                    }
                } else {
                    Log.w(TAG, "Qwen3 startDecode failed")
                    engine.reset()
                    // Restart listening on decode failure
                    lifecycleScope.launch {
                        kotlinx.coroutines.delay(500)
                        if (!isStopped) startQwen3Record()
                    }
                }
            }
        } else {
            // No speech detected — restart listening
            Log.d(TAG, "Qwen3: no speech detected, restarting record")
            lifecycleScope.launch {
                kotlinx.coroutines.delay(200)
                if (!isStopped) startQwen3Record()
            }
        }
    }

    private fun stopQwen3Record() {
        if (qwen3IsRecording.get()) {
            qwen3IsRecording.set(false)
            isRecording = false
            try { qwen3Aec?.release() } catch (_: Exception) {}
            qwen3Aec = null
            try { qwen3Ns?.release() } catch (_: Exception) {}
            qwen3Ns = null
            try {
                qwen3AudioRecord?.stop()
                qwen3AudioRecord?.release()
            } catch (_: Exception) {}
            qwen3AudioRecord = null
            qwen3RecordingThread = null
            Log.d(TAG, "Qwen3 recording stopped")
        }
    }

    /**
     * Write PCM float samples to a WAV file (16-bit mono, 16kHz).
     * Used by Omni audio mode to save recording for <audio> tag input.
     */
    private fun writeWavFile(samples: FloatArray, sampleRate: Int, filePath: String): Boolean {
        return try {
            val wavFile = File(filePath)
            val dataSize = samples.size * 2  // 16-bit = 2 bytes per sample
            val buffer = ByteBuffer.allocate(44 + dataSize)
            buffer.order(ByteOrder.LITTLE_ENDIAN)

            // RIFF header
            buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
            buffer.putInt(36 + dataSize)       // File size - 8
            buffer.put("WAVE".toByteArray(Charsets.US_ASCII))
            // fmt chunk
            buffer.put("fmt ".toByteArray(Charsets.US_ASCII))
            buffer.putInt(16)                   // Subchunk1Size (PCM = 16)
            buffer.putShort(1)                  // Audio format (1 = PCM)
            buffer.putShort(1)                  // NumChannels (mono)
            buffer.putInt(sampleRate)           // Sample rate
            buffer.putInt(sampleRate * 2)       // Byte rate
            buffer.putShort(2)                  // Block align
            buffer.putShort(16)                 // Bits per sample
            // data chunk
            buffer.put("data".toByteArray(Charsets.US_ASCII))
            buffer.putInt(dataSize)
            // PCM float32 → int16
            for (sample in samples) {
                val clamped = (sample * 32767f).toInt().coerceIn(-32768, 32767)
                buffer.putShort(clamped.toShort())
            }

            FileOutputStream(wavFile).use { it.write(buffer.array()) }
            val durationSec = "%.1f".format(samples.size.toFloat() / sampleRate)
            Log.i(TAG, "WAV written: ${wavFile.absolutePath} (${samples.size} samples, ${durationSec}s)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write WAV file: ${filePath}", e)
            false
        }
    }

    private fun llmGenerate(text: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            Log.d(TAG, "Starting LLM generation... isStopped: $isStopped")
            if (isStopped) return@launch

            // Reset generation state
            responseBuilder.clear()
            ttsSegmentBuffer.clear()
            isFirstChunk = true
            isGenerationFinished = false

            // Send message through ChatPresenter for proper session management
            chatPresenter.sendMessage(text)
        }
    }

    private fun stopRecord() {
        if (asrMode != AsrMode.SHERPA) {
            stopQwen3Record()
        } else {
            if (isRecording) {
                asrService?.stopRecord()
                isRecording = false
                Log.d(TAG, "Recording stopped")
            }
        }
    }

    private fun startRecord() {
        if (!isRecording && !isSpeaking && !isProcessingLlm) {
            if (asrMode != AsrMode.SHERPA) {
                startQwen3Record()
            } else {
                asrService?.startRecord()
                isRecording = true
                Log.d(TAG, "Recording started")
            }
        }
    }

    fun getCurrentStatus(): VoiceChatPresenterState {
        return currentStatus
    }

    private fun speakGreetingMessage() {
        lifecycleScope.launch {
            try {
                if (isStopped) return@launch
                
                // Get the greeting message from resources (Android will auto-select language)
                val greetingMessage = activity.getString(com.alibaba.mnnllm.android.R.string.voice_chat_ready_greeting)
                
                // We don't call `stopRecord()` here to keep ASR recording active to allow user to skip or interrupt the greeting.
                // stopRecord()
                
                // Set status to greeting
                currentStatus = VoiceChatPresenterState.PLAYING
                withContext(Dispatchers.Main) {
                    view.updateStatus(VoiceChatState.GREETING)
                }

                // Automatically mute during greeting if Auto-Mute mode is enabled
                if (isAutoMuteForEchoCancelMode) {
                    muteMicrophone(true)
                }
                
                // Generate TTS audio for greeting
                withContext(Dispatchers.IO) {
                    if (isStopped) return@withContext
                    
                    Log.d(TAG, "Speaking greeting message: $greetingMessage")
                    val audioData = processTtsText(greetingMessage)
                    
                    if (audioData != null && audioData.isNotEmpty() && !isStopped) {
                        withContext(Dispatchers.Main) {
                            // Store the original listener
                            val originalListener = {
                                Log.d(TAG, "Audio playback completed - currentStatus: ${currentStatus.name}, isSpeaking: $isSpeaking, isProcessingLlm: $isProcessingLlm")
                                currentStatus = VoiceChatPresenterState.PLAY_END
                                lifecycleScope.launch {
                                    Log.d(TAG, "Sending OnTtsComplete task")
                                    taskChannel.send(SerialTask.OnTtsComplete)
                                }
                                Unit // Explicitly return Unit to fix compilation error
                            }
                            
                            // Set up temporary completion listener for greeting
                            audioPlayer?.setOnCompletionListener {
                                Log.d(TAG, "Greeting message playback completed")
                                lifecycleScope.launch {
                                    // Resume normal state after greeting
                                    currentStatus = VoiceChatPresenterState.LISTENING
                                    withContext(Dispatchers.Main) {
                                        view.updateStatus(VoiceChatState.LISTENING)
                                    }
                                    // Small delay then resume recording
                                    kotlinx.coroutines.delay(300)

                                    // Automatically un-mute after greeting if Auto-Mute mode is enabled
                                    if (isAutoMuteForEchoCancelMode) {
                                        muteMicrophone(false)
                                    }

                                    startRecord()
                                    
                                    // Restore the original completion listener for normal TTS
                                    // Do this on the main thread to avoid threading issues
                                    withContext(Dispatchers.Main) {
                                        audioPlayer?.reset()
                                        audioPlayer?.setOnCompletionListener(originalListener)
                                        Log.d(TAG, "Original completion listener restored")
                                    }
                                }
                            }
                            
                            // Play the greeting audio
                            audioPlayer?.playChunk(audioData)
                            audioPlayer?.endChunk()
                        }
                    } else {
                        Log.w(TAG, "Failed to generate TTS audio for greeting message")
                        // If TTS fails, just resume recording
                        withContext(Dispatchers.Main) {
                            currentStatus = VoiceChatPresenterState.LISTENING
                            view.updateStatus(VoiceChatState.LISTENING)
                        }
                        if (isAutoMuteForEchoCancelMode) {
                            muteMicrophone(false)
                        }
                        startRecord()
                    }
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error speaking greeting message", e)
                // On error, just resume normal state
                currentStatus = VoiceChatPresenterState.LISTENING
                withContext(Dispatchers.Main) {
                    view.updateStatus(VoiceChatState.LISTENING)
                }
                if (isAutoMuteForEchoCancelMode) {
                    muteMicrophone(false)
                }
                startRecord()
            }
        }
    }

    private suspend fun processTtsText(text: String): ShortArray? {
        val service = ttsService ?: return null
        val isReady = service.waitForInitComplete()
        if (!isReady) {
            Log.w(TAG, "TTS Service not ready, skipping synthesis for text: $text")
            return null
        }
        return service.process(text, 0)
    }

    fun stop() {
        Log.d(TAG, "Presenter stopping...")
        isStopped = true
        
        // Reset generation state
        isGenerationFinished = false
        
        // Stop any ongoing generation and trigger ChatActivity's stop logic
        if (isProcessingLlm || isSpeaking) {
            chatPresenter.stopGenerate()
            if (activity is com.alibaba.mnnllm.android.chat.ChatActivity) {
                activity.onStopGenerationRequested()
            }
        }
        
        // Unregister from ChatPresenter
        chatPresenter.removeGenerateListener(this)
        
        if (isRecording) {
            try {
                if (asrMode == AsrMode.QWEN3_OLD) {
                    stopQwen3Record()
                    qwen3AsrEngine?.release()
                    qwen3AsrEngine = null
                } else if (asrMode == AsrMode.QWEN3_OMNI) {
                    stopQwen3Record()
                    omniAudioBuffer.clear()
                } else {
                    asrService?.stopRecord()
                    asrService = null
                }
                isRecording = false
                Log.d(TAG, "ASR record stopped.")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping ASR record", e)
            }
        }
        try {
            audioPlayer?.destroy()
            ttsService?.destroy()
            ttsService = null
            audioPlayer = null
            Log.d(TAG, "TTS and AudioPlayer destroyed.")
        } catch (e: Exception) {
            Log.e(TAG, "Error destroying TTS service", e)
        }
        
        // Cleanup serial processor
        try {
            taskChannel.close()
            Log.d(TAG, "Serial processor closed.")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing serial processor", e)
        }
    }

    fun toggleSpeaker(isSpeakerOn: Boolean) {
        audioManager.isSpeakerphoneOn = isSpeakerOn
        Log.d(TAG, "Speaker toggled: $isSpeakerOn")
    }

    fun toggleMute() {
        muteMicrophone(!isMuted)
    }

    private fun muteMicrophone(mute: Boolean) {
        if (isMuted != mute) {
            isMuted = mute
            if (asrMode == AsrMode.SHERPA) {
                asrService?.setMuted(isMuted)
            }
            // For Qwen3 modes (OLD and OMNI), isMuted flag is checked directly in
            // processQwen3Samples() — the shared recording loop handles muting via shortBuf.fill(0)
            view.updateMuteButtonState(isMuted)
            Log.d(TAG, "Microphone mute state changed: $isMuted (asrMode=$asrMode)")
        }
    }

    fun toggleEchoCancelMode() {
        isAutoMuteForEchoCancelMode = !isAutoMuteForEchoCancelMode
        view.updateEchoCancelMode(isAutoMuteForEchoCancelMode)
        Log.d(TAG, "Echo cancel mode toggled, auto mute: $isAutoMuteForEchoCancelMode")
    }

    fun stopGeneration() {
        Log.d(TAG, "Stopping generation...")
        if (isProcessingLlm || isSpeaking) {
            isStoppingGeneration = true
            isGenerationFinished = false
            
            // Stop generation in ChatPresenter
            chatPresenter.stopGenerate()
            
            // Trigger ChatActivity's stop logic
            if (activity is com.alibaba.mnnllm.android.chat.ChatActivity) {
                activity.onStopGenerationRequested()
            }
            
            audioPlayer?.stop()
            isProcessingLlm = false
            isSpeaking = false
            currentStatus = VoiceChatPresenterState.LISTENING
            
            lifecycleScope.launch {
                withContext(Dispatchers.Main) {
                    view.updateStatus(VoiceChatState.STOPPING)
                }
                // Small delay to show stopping state
                kotlinx.coroutines.delay(300)
                withContext(Dispatchers.Main) {
                    view.updateStatus(VoiceChatState.LISTENING)
                }
                // Reset audio player and restart recording
                audioPlayer?.reset()
                kotlinx.coroutines.delay(200)

                // Ensure mic is un-muted when stopping generation manually
                if (isAutoMuteForEchoCancelMode) {
                    muteMicrophone(false)
                }

                isStoppingGeneration = false
                startRecord()
            }
        }
    }
    
    /**
     * Recreate ASR and TTS services with new models
     * This method should be called when the default voice models have changed
     */
    fun recreateVoiceServices() {
        Log.d(TAG, "Recreating voice services due to model changes...")
        
        lifecycleScope.launch {
            try {
                // Stop current services
                stopRecord()
                
                // Reset generation state
                isGenerationFinished = false
                
                // Cleanup existing services
                if (asrMode == AsrMode.QWEN3_OLD) {
                    stopQwen3Record()
                    qwen3AsrEngine?.release()
                    qwen3AsrEngine = null
                } else if (asrMode == AsrMode.QWEN3_OMNI) {
                    stopQwen3Record()
                    omniAudioBuffer.clear()
                } else {
                    asrService?.stopRecord()
                    asrService = null
                }
                asrMode = AsrMode.SHERPA  // Will be re-detected in startAsr()
                
                ttsService?.destroy()
                ttsService = null
                
                // Show connecting state
                currentStatus = VoiceChatPresenterState.INITIALIZING
                withContext(Dispatchers.Main) {
                    view.updateStatus(VoiceChatState.CONNECTING)
                }
                
                // Reinitialize services with new models
                initTts()
                startAsr()
                
                Log.d(TAG, "Voice services recreated successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error recreating voice services", e)
                if (!isStopped) {
                    withContext(Dispatchers.Main) {
                        view.showError("Failed to recreate voice services: ${e.message}")
                    }
                }
            }
        }
    }
    
    // ChatPresenter.GenerateListener implementation
    override fun onGenerateStart() {
        // No additional action needed for voice chat UI
    }
    
    override fun onLlmGenerateProgress(progress: String?, generateResultProcessor: GenerateResultProcessor) {
        if (isStopped || isStoppingGeneration || progress == null) return
        
        lifecycleScope.launch {
            if (isStopped || isStoppingGeneration) return@launch
            
            if (isFirstChunk) {
                taskChannel.send(SerialTask.ProcessProgress(progress, true, responseBuilder, ttsSegmentBuffer))
                isFirstChunk = false
            } else {
                taskChannel.send(SerialTask.ProcessProgress(progress, false, responseBuilder, ttsSegmentBuffer))
            }
        }
    }
    
    override fun onDiffusionGenerateProgress(progress: String?, diffusionDestPath: String?) {
        // Not used in voice chat
    }
    
    override fun onGenerateFinished(benchMarkResult: HashMap<String, Any>) {
        if (isStopped || isStoppingGeneration) return
        
        if (isGenerationFinished) {
            Log.d(TAG, "onGenerateFinished already processed, ignoring duplicate call")
            return
        }
        
        isGenerationFinished = true
        Log.d(TAG, "onGenerateFinished called, sending ProcessFinalChunk task")
        
        lifecycleScope.launch {
            if (!isStoppingGeneration) {
                taskChannel.send(SerialTask.ProcessFinalChunk(ttsSegmentBuffer))
            }
        }
    }
}

interface VoiceChatView {
    fun updateStatus(state: VoiceChatState)
    fun addTranscript(transcript: Transcript)
    fun updateLastTranscript(text: String)
    fun showError(message: String)
    fun stopGeneration()
    fun showGreetingMessage()
    fun updateMuteButtonState(isMuted: Boolean)
    fun updateEchoCancelMode(isAutoMuteForEchoCancelMode: Boolean)
    fun capturePhoto()
    fun getCapturedImageUri(): android.net.Uri?
    fun clearCapturedImageUri()
    fun isCameraEnabled(): Boolean
    // Qwen3-ASR streaming partial result (updates live during decoding)
    fun updateAsrPartialText(text: String) {}
}

interface TtsClient {
    fun setLanguage(language: String)
    suspend fun init(modelDir: String): Boolean
    suspend fun waitForInitComplete(): Boolean
    fun process(text: String, id: Int): ShortArray
    fun destroy()
}

class RealTtsClient(private val service: TtsService) : TtsClient {
    override fun setLanguage(language: String) = service.setLanguage(language)

    override suspend fun init(modelDir: String): Boolean = service.init(modelDir)

    override suspend fun waitForInitComplete(): Boolean = service.waitForInitComplete()

    override fun process(text: String, id: Int): ShortArray = service.process(text, id)

    override fun destroy() = service.destroy()
}
