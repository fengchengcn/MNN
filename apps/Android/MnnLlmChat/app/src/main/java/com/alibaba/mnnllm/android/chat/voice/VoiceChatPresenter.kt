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
import java.io.File
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
    THINKING,
    SPEAKING,
    STOPPING,
    ERROR
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
    private var qwen3RecordingThread: Thread? = null
    private var isQwen3Mode = false
    private val qwen3IsRecording = AtomicBoolean(false)
    // Silence-based endpoint detection for Qwen3-ASR
    private var silenceChunkCount = 0
    private var speechDetectedForQwen3 = false
    private val silenceRmsThreshold = 100.0f   // RMS below this = silence
    private val speechRmsThreshold = 400.0f    // RMS above this = speech (for interruption)
    private val maxSilenceChunks = 15           // ~1.5s silence triggers endpoint
    private val qwen3SampleRate = 16000

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
                    view.addTranscript(Transcript(isUser = true, text = task.text))
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
     * Detect whether the model directory is a Qwen3-ASR model
     * (by checking for the presence of audio_encoder.mnn).
     */
    private fun isQwen3AsrModel(modelDir: String): Boolean {
        val markerFile = File(modelDir, "audio_encoder.mnn")
        val result = markerFile.exists()
        Log.d(TAG, "ASR model type check: $modelDir → isQwen3=$result")
        return result
    }

    private fun startAsr() {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                if (isStopped) return@launch

                Log.d(TAG, "Initializing ASR Service...")
                val modelDir = VoiceModelPathUtils.getAsrModelPath(activity)
                Log.i(TAG, "Using ASR model path: $modelDir")

                // --- Detect model type ---
                isQwen3Mode = isQwen3AsrModel(modelDir)

                if (isQwen3Mode) {
                    startQwen3Asr(modelDir)
                } else {
                    startSherpaAsr(modelDir)
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
                val aec = AcousticEchoCanceler.create(qwen3AudioRecord!!.audioSessionId)
                aec.enabled = true
                Log.i(TAG, "Qwen3: AEC enabled")
            }
        } catch (_: Exception) {}
        try {
            if (NoiseSuppressor.isAvailable()) {
                val ns = NoiseSuppressor.create(qwen3AudioRecord!!.audioSessionId)
                ns.enabled = true
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
        val engine = qwen3AsrEngine ?: return

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

            // Convert int16 → float32 [-1, 1]
            val floatBuf = FloatArray(ret) { i -> shortBuf[i] / 32768.0f }

            // RMS energy for silence/speech detection
            var sumSq = 0.0f
            for (s in floatBuf) sumSq += s * s
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
            engine.pushAudio(floatBuf)

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

        // Run decoder if we had speech
        if (speechDetectedForQwen3) {
            Log.i(TAG, "Qwen3 running decoder...")
            engine.endAudio()
            val text = engine.getResultText()
            Log.i(TAG, "Qwen3 ASR result: $text")
            engine.reset()

            if (text.isNotEmpty() && !isStopped && !isSpeaking && !isProcessingLlm) {
                lifecycleScope.launch {
                    taskChannel.send(SerialTask.HandleAsrResult(text))
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
            try {
                qwen3AudioRecord?.stop()
                qwen3AudioRecord?.release()
            } catch (_: Exception) {}
            qwen3AudioRecord = null
            qwen3RecordingThread = null
            Log.d(TAG, "Qwen3 recording stopped")
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
        if (isQwen3Mode) {
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
            if (isQwen3Mode) {
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
                if (isQwen3Mode) {
                    stopQwen3Record()
                    qwen3AsrEngine?.release()
                    qwen3AsrEngine = null
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
            if (!isQwen3Mode) {
                asrService?.setMuted(isMuted)
            }
            // For Qwen3, isMuted flag is checked directly in processQwen3Samples()
            view.updateMuteButtonState(isMuted)
            Log.d(TAG, "Microphone mute state changed: $isMuted (qwen3Mode=$isQwen3Mode)")
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
                if (isQwen3Mode) {
                    stopQwen3Record()
                    qwen3AsrEngine?.release()
                    qwen3AsrEngine = null
                    isQwen3Mode = false
                } else {
                    asrService?.stopRecord()
                    asrService = null
                }
                
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
