package com.alibaba.mnnllm.android.asr

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import android.os.Bundle
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.animation.AlphaAnimation
import android.view.animation.Animation
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.Timer
import java.util.TimerTask
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

enum class TestMode { BATCH, STREAMING, OMNI }

class Qwen3AsrTestActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "Qwen3AsrTest"
        private const val SAMPLE_RATE = 16000
        private const val REQUEST_AUDIO = 100

        // Silence detection thresholds (reused from VoiceChatPresenter)
        private const val SPEECH_RMS_THRESHOLD = 400.0f
        private const val SILENCE_RMS_THRESHOLD = 100.0f
        private const val MAX_SILENCE_CHUNKS = 15
        private const val MAX_TOTAL_CHUNKS = 300
        private const val CHUNK_INTERVAL_MS = 100
    }

    // ── UI ──
    private lateinit var chipBatch: TextView
    private lateinit var chipStreaming: TextView
    private lateinit var chipOmni: TextView
    private lateinit var btnRecord: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvEmptyResults: TextView
    private lateinit var btnClear: TextView
    private lateinit var audioLevelFill: View
    private lateinit var audioLevelContainer: LinearLayout
    private lateinit var resultsContainer: LinearLayout

    // ── State ──
    @Volatile private var currentMode = TestMode.BATCH
    private var engine: Qwen3AsrEngine? = null
    private var llmSession: LlmSession? = null          // Omni path
    private var omniModelDir: String? = null             // Omni model config directory
    private var audioRecord: AudioRecord? = null
    private var aec: AcousticEchoCanceler? = null
    private var noiseSuppressor: NoiseSuppressor? = null
    private val isRecording = AtomicBoolean(false)
    private val stoppedByUser = AtomicBoolean(false)
    private var recordingThread: Thread? = null
    private var resultCardCount = 0
    private val timeFormatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    // ── Streaming / Omni state ──
    @Volatile private var silenceChunkCount = 0
    @Volatile private var speechDetected = false
    @Volatile private var currentRms = 0f
    private val omniAudioBuffer = mutableListOf<Float>()  // Omni: accumulate PCM float samples

    // ── Phase 2: Sliding-window pseudo-streaming state ──
    private var streamingTimer: Timer? = null
    private var lastProcessedLen = 0
    private var recordingStartTime = 0L
    @Volatile private var streamingIncrementalInProgress = false
    private var streamingResultCard: LinearLayout? = null
    private var streamingResultTextView: TextView? = null

    // ── Blink animation ──
    private var blinkAnimation: AlphaAnimation? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_qwen3_asr_test)

        // Bind views
        chipBatch = findViewById(R.id.chipBatch)
        chipStreaming = findViewById(R.id.chipStreaming)
        chipOmni = findViewById(R.id.chipOmni)
        btnRecord = findViewById(R.id.btnRecord)
        tvStatus = findViewById(R.id.tvStatus)
        tvEmptyResults = findViewById(R.id.tvEmptyResults)
        btnClear = findViewById(R.id.btnClear)
        audioLevelFill = findViewById(R.id.audioLevelFill)
        audioLevelContainer = findViewById(R.id.audioLevelContainer)
        resultsContainer = findViewById(R.id.resultsContainer)

        blinkAnimation = AlphaAnimation(0.3f, 1.0f).apply {
            duration = 600
            repeatMode = Animation.REVERSE
            repeatCount = Animation.INFINITE
        }

        // Request permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_AUDIO)
        }

        // Load native lib
        try {
            System.loadLibrary("mnnllmapp")
            appendSystemMessage("Native library loaded OK")
        } catch (e: UnsatisfiedLinkError) {
            setStatus("FATAL: Library load failed")
            appendSystemMessage("System.loadLibrary(\"mnnllmapp\") failed: ${e.message}")
            btnRecord.isEnabled = false
            return
        }

        // Init engine
        setStatus("Initializing engine...")
        btnRecord.isEnabled = false
        lifecycleScope.launch(Dispatchers.IO) { initEngine() }

        // Listeners
        chipBatch.setOnClickListener { switchMode(TestMode.BATCH) }
        chipStreaming.setOnClickListener { switchMode(TestMode.STREAMING) }
        chipOmni.setOnClickListener { switchMode(TestMode.OMNI) }
        btnRecord.setOnClickListener {
            if (isRecording.get()) {
                stopRecording()
            } else {
                startRecording()
            }
        }
        btnClear.setOnClickListener { clearResults() }
    }

    // ══════════════════════════════════════════════
    //  Engine Init
    // ══════════════════════════════════════════════

    /**
     * Scan /data/local/tmp/mnn_models/ for Omni-compatible audio models.
     * Returns the config directory path if found.
     */
    private fun findOmniModel(): String? {
        val localDir = File("/data/local/tmp/mnn_models")
        if (!localDir.exists() || !localDir.isDirectory) return null
        localDir.listFiles()?.forEach { subdir ->
            if (!subdir.isDirectory) return@forEach
            val audioMnn = File(subdir, "audio.mnn")
            val configJson = File(subdir, "config.json")
            if (audioMnn.exists() && configJson.exists()) {
                try {
                    val config = JSONObject(configJson.readText())
                    if (config.optBoolean("is_audio", false)) {
                        Log.i(TAG, "Found Omni model: ${subdir.absolutePath}")
                        return subdir.absolutePath
                    }
                } catch (_: Exception) {}
            }
        }
        return null
    }

    private suspend fun initEngine() {
        try {
            // ── Omni model detection + loading ──
            val omniDir = findOmniModel()
            var omniReady = false
            if (omniDir != null) {
                Log.i(TAG, "Omni model dir: $omniDir")
                withContext(Dispatchers.Main) {
                    setStatus("Omni model found — loading...")
                }

                try {
                    val configPath = "$omniDir/config.json"
                    llmSession = ChatService.provide().createLlmSession(
                        "omni_test",
                        configPath,
                        "omni_test_${System.currentTimeMillis()}",
                        null,          // no history
                        true,          // supportOmni = true
                        "cpu"          // backendType
                    ) as? LlmSession
                    llmSession?.load()
                    Log.i(TAG, "Omni LlmSession loaded OK")
                    omniModelDir = omniDir
                    omniReady = true
                    withContext(Dispatchers.Main) {
                        appendSystemMessage("Omni loaded: $omniDir")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Omni LlmSession load failed", e)
                    llmSession = null
                    withContext(Dispatchers.Main) {
                        appendSystemMessage("Omni load FAILED: ${e.message}")
                    }
                }
            }

            // ── Determine final state ──
            withContext(Dispatchers.Main) {
                if (omniReady) {
                    setStatus("Omni ready — tap REC to test")
                    chipBatch.visibility = View.GONE
                    chipStreaming.visibility = View.GONE
                    chipOmni.visibility = View.VISIBLE
                    chipOmni.performClick()  // auto-select Omni mode
                    btnRecord.isEnabled = true
                } else {
                    setStatus("ERROR: Omni not available")
                    chipBatch.visibility = View.GONE
                    chipStreaming.visibility = View.GONE
                    chipOmni.visibility = View.GONE
                    appendSystemMessage("Place model at /data/local/tmp/mnn_models/Qwen3-ASR-MNN-INT8/")
                    appendSystemMessage("Ensure audio.mnn + config.json (is_audio=true) exist")
                    btnRecord.isEnabled = false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Engine init error", e)
            withContext(Dispatchers.Main) { setStatus("ERROR: ${e.message}") }
        }
    }


    // ══════════════════════════════════════════════
    //  Mode Switching
    // ══════════════════════════════════════════════

    private fun switchMode(mode: TestMode) {
        if (isRecording.get()) return
        currentMode = mode

        // Reset all chips
        val dimColor = Color.parseColor("#888899")
        chipBatch.setBackgroundResource(R.drawable.bg_mode_chip_normal)
        chipBatch.setTextColor(dimColor)
        chipStreaming.setBackgroundResource(R.drawable.bg_mode_chip_normal)
        chipStreaming.setTextColor(dimColor)
        chipOmni.setBackgroundResource(R.drawable.bg_mode_chip_normal)
        chipOmni.setTextColor(dimColor)

        btnRecord.text = "REC"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
        audioLevelContainer.visibility = View.GONE

        when (mode) {
            TestMode.BATCH -> {
                chipBatch.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipBatch.setTextColor(Color.WHITE)
                setStatus("Batch mode — tap REC to start")
            }
            TestMode.STREAMING -> {
                chipStreaming.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipStreaming.setTextColor(Color.WHITE)
                setStatus("Streaming mode — auto endpoint detection")
            }
            TestMode.OMNI -> {
                chipOmni.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipOmni.setTextColor(Color.WHITE)
                setStatus("Omni mode — LLM-powered ASR (WAV → Omni engine)")
            }
        }
    }

    // ══════════════════════════════════════════════
    //  Recording — Entry Points
    // ══════════════════════════════════════════════

    private fun startRecording() {
        if (isRecording.get()) return

        engine?.reset()
        omniAudioBuffer.clear()
        isRecording.set(true)
        stoppedByUser.set(false)
        silenceChunkCount = 0
        speechDetected = false
        currentRms = 0f

        btnRecord.text = "STOP"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_active)

        when (currentMode) {
            TestMode.BATCH -> setStatus("● Recording... tap STOP when done")
            TestMode.STREAMING -> {
                setStatus("● Listening...")
                tvStatus.startAnimation(blinkAnimation)
            }
            TestMode.OMNI -> {
                setStatus("● Recording (Stream)... tap STOP when done")
                lastProcessedLen = 0
                recordingStartTime = System.currentTimeMillis()
                streamingIncrementalInProgress = false
                // Remove any stale streaming card from a previous session
                streamingResultCard?.let { resultsContainer.removeView(it) }
                streamingResultCard = null
                streamingResultTextView = null
                startStreamingTimer()
            }
        }

        // Init audio
        initAudioRecord()
        audioRecord?.startRecording()

        val chunkSize = (CHUNK_INTERVAL_MS * SAMPLE_RATE / 1000).toInt()
        recordingThread = Thread { recordingLoop(chunkSize) }
        recordingThread?.start()
    }

    private fun stopRecording() {
        if (currentMode == TestMode.STREAMING) {
            // Signal stop; the recording thread will handle processing naturally,
            // then check stoppedByUser to decide whether to auto-restart.
            stoppedByUser.set(true)
        }
        // Phase 2: Cancel streaming timer (Omni mode)
        stopStreamingTimer()
        isRecording.set(false)
        tvStatus.clearAnimation()
    }

    // ══════════════════════════════════════════════
    //  Recording Loop (runs on background thread)
    // ══════════════════════════════════════════════

    private fun recordingLoop(chunkSize: Int) {
        val shortBuf = ShortArray(chunkSize)
        var totalChunks = 0
        val isStreaming = (currentMode == TestMode.STREAMING)
        val isOmni = (currentMode == TestMode.OMNI)

        // Read loop
        while (isRecording.get() && audioRecord != null && totalChunks < MAX_TOTAL_CHUNKS) {
            totalChunks++
            val ret = audioRecord?.read(shortBuf, 0, chunkSize) ?: 0
            if (ret <= 0) continue

            val floatBuf = FloatArray(ret) { i -> shortBuf[i] / 32768.0f }

            if (isOmni) {
                // Omni mode: collect audio samples for WAV writing
                synchronized(omniAudioBuffer) {
                    omniAudioBuffer.addAll(floatBuf.asList())
                }
            } else {
                engine?.pushAudio(floatBuf)
            }

            // RMS on raw int16 values (thresholds are in raw PCM scale)
            var sumSq = 0f
            for (i in 0 until ret) {
                val s = shortBuf[i].toFloat()
                sumSq += s * s
            }
            val rms = sqrt(sumSq / ret)
            currentRms = rms
            runOnUiThread { updateAudioLevel(rms) }

            if (isStreaming) {
                // VAD logic
                if (rms > SPEECH_RMS_THRESHOLD) {
                    if (!speechDetected) {
                        speechDetected = true
                        runOnUiThread {
                            tvStatus.clearAnimation()
                            setStatus("● Speech detected...")
                        }
                    }
                    silenceChunkCount = 0
                } else if (speechDetected && rms < SILENCE_RMS_THRESHOLD) {
                    silenceChunkCount++
                } else if (rms >= SILENCE_RMS_THRESHOLD) {
                    silenceChunkCount = 0
                }

                // Endpoint check
                if (speechDetected && silenceChunkCount >= MAX_SILENCE_CHUNKS) {
                    Log.i(TAG, "Endpoint: $silenceChunkCount silence chunks")
                    break
                }
            }
        }

        if (totalChunks >= MAX_TOTAL_CHUNKS) {
            Log.w(TAG, "Max duration reached")
            speechDetected = true
        }

        // Clean up audio hardware
        stopAudioHardware()
        isRecording.set(false)

        if (isOmni) {
            // ── Phase 2: Stop streaming timer, run final full-buffer inference ──
            stopStreamingTimer()

            val samples = synchronized(omniAudioBuffer) { omniAudioBuffer.toList() }
            omniAudioBuffer.clear()
            if (samples.isEmpty()) {
                runOnUiThread { returnToIdle() }
                return
            }
            runOnUiThread {
                btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
                btnRecord.text = "..."
                btnRecord.isEnabled = false
                setStatus("Final decoding...")
            }
            processIncrementalOmni(samples.toFloatArray(), isFinal = true)
        } else if (isStreaming) {
            // Decide what to do based on speech and user intent
            if (speechDetected) {
                runOnUiThread {
                    // Keep button interactive in streaming mode; only update status text
                    setStatus("Decoding...")
                    tvStatus.clearAnimation()
                }
                decodeAndHandleStreamingResult()
            } else {
                // No speech detected yet
                if (stoppedByUser.get()) {
                    runOnUiThread { returnToIdle() }
                } else {
                    runOnUiThread {
                        setStatus("No speech — listening...")
                    }
                    restartStreamingIfNeeded()
                }
            }
        } else {
            // Batch mode: always decode
            runOnUiThread {
                btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
                btnRecord.text = "..."
                btnRecord.isEnabled = false
                setStatus("Decoding...")
            }
            engine?.endAudio()
            val text = engine?.getResultText() ?: ""
            runOnUiThread { onBatchResult(text) }
        }
    }

    // ══════════════════════════════════════════════
    // ══════════════════════════════════════════════
    //  Omni: Write WAV and Send to LlmSession
    // ══════════════════════════════════════════════

    /** Write 16-bit little-endian short to RandomAccessFile */
    private fun RandomAccessFile.writeShortLE(value: Int) {
        val v = value and 0xFFFF
        write(v and 0xFF)           // low byte first
        write((v ushr 8) and 0xFF)  // high byte
    }

    private fun writeWavFile(samples: FloatArray, sampleRate: Int, filePath: String): Boolean {
        return try {
            val numChannels = 1
            val bitsPerSample = 16
            val byteRate = sampleRate * numChannels * bitsPerSample / 8
            val blockAlign = numChannels * bitsPerSample / 8
            val dataSize = samples.size * blockAlign
            val fileSize = 36 + dataSize

            val file = RandomAccessFile(filePath, "rw")
            file.setLength(0)

            // RIFF header
            file.writeBytes("RIFF")
            file.writeInt(Integer.reverseBytes(fileSize))
            file.writeBytes("WAVE")

            // fmt subchunk
            file.writeBytes("fmt ")
            file.writeInt(Integer.reverseBytes(16))  // subchunk1 size (32-bit)
            file.writeShortLE(1)   // PCM format (16-bit LE)
            file.writeShortLE(numChannels)
            file.writeInt(Integer.reverseBytes(sampleRate))
            file.writeInt(Integer.reverseBytes(byteRate))
            file.writeShortLE(blockAlign)
            file.writeShortLE(bitsPerSample)

            // data subchunk
            file.writeBytes("data")
            file.writeInt(Integer.reverseBytes(dataSize))

            // PCM samples (float → int16)
            val byteBuf = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (s in samples) {
                val intSample = (s * 32767f).toInt().coerceIn(-32768, 32767)
                byteBuf.putShort(intSample.toShort())
            }
            file.write(byteBuf.array())
            file.close()
            Log.i(TAG, "WAV written: $filePath (${dataSize} bytes, ${samples.size} samples)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error writing WAV: $filePath", e)
            false
        }
    }

    // ══════════════════════════════════════════════
    //  Phase 2: Streaming Timer & Incremental Omni
    // ══════════════════════════════════════════════

    private fun startStreamingTimer() {
        streamingTimer = Timer("omni-stream", true).apply {
            scheduleAtFixedRate(object : TimerTask() {
                override fun run() {
                    if (!isRecording.get() || streamingIncrementalInProgress) return

                    val snapshot = synchronized(omniAudioBuffer) { omniAudioBuffer.toFloatArray() }
                    val newSamples = snapshot.size - lastProcessedLen
                    // Skip if less than 1 second of new audio (avoids redundant inference)
                    if (newSamples < SAMPLE_RATE) return

                    lastProcessedLen = snapshot.size
                    streamingIncrementalInProgress = true

                    val elapsed = (System.currentTimeMillis() - recordingStartTime) / 1000f
                    runOnUiThread {
                        setStatus("Streaming — recorded %.1fs, processing...".format(elapsed))
                    }
                    Log.i(TAG, "Phase2 incremental: ${snapshot.size} samples, ${newSamples} new")
                    processIncrementalOmni(snapshot, isFinal = false)
                }
            }, 2000, 2000)  // initial delay 2s, then every 2s
        }
    }

    private fun stopStreamingTimer() {
        streamingTimer?.cancel()
        streamingTimer = null
    }

    /**
     * Send audio to Omni engine and handle progress. Used for both:
     * - Incremental snapshots during recording (isFinal=false): updates live streaming card
     * - Final full-buffer inference at STOP (isFinal=true): finalizes result and returns to idle
     */
    private fun processIncrementalOmni(samples: FloatArray, isFinal: Boolean) {
        if (llmSession == null) {
            Log.w(TAG, "Phase2: LlmSession is null, skipping")
            streamingIncrementalInProgress = false
            if (isFinal) runOnUiThread { returnToIdle() }
            return
        }

        llmSession?.setAudioData(samples, SAMPLE_RATE)
        val audioTag = "<audio>stream</audio>"
        Log.i(TAG, "Phase2 ${if (isFinal) "FINAL" else "incr"}: $audioTag, ${samples.size} samples")

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                var fullText = ""
                llmSession?.generate(audioTag, mapOf(), object : GenerateProgressListener {
                    override fun onProgress(progress: String?): Boolean {
                        if (progress != null) {
                            fullText += progress
                            runOnUiThread { updateStreamingResult(fullText) }
                        }
                        return false  // don't cancel
                    }
                })

                val response = fullText
                Log.i(TAG, "Phase2 ${if (isFinal) "FINAL" else "incr"} result: $response")

                withContext(Dispatchers.Main) {
                    if (isFinal) {
                        finalizeStreamingResult(response)
                    } else {
                        // Incremental done — update status with elapsed time
                        if (isRecording.get()) {
                            val elapsed = (System.currentTimeMillis() - recordingStartTime) / 1000f
                            setStatus("Streaming — recorded %.1fs".format(elapsed))
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Phase2 Omni error", e)
                withContext(Dispatchers.Main) {
                    if (isFinal) {
                        appendSystemMessage("OMNI ERROR: ${e.message}")
                        returnToIdle()
                    }
                }
            } finally {
                streamingIncrementalInProgress = false
            }
        }
    }

    // ── Streaming Result UI ──

    /** Create or get the live streaming result card (idempotent) */
    private fun ensureStreamingCard() {
        if (streamingResultCard != null) return

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_asr_result_card)
            val pad = dp(16)
            setPadding(pad, pad, pad, pad)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(12) }
        }

        // Header: LIVE badge + elapsed time
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(8) }
        }

        val liveBadge = TextView(this).apply {
            setText("● LIVE")
            setTextColor(Color.parseColor("#FF4444"))
            textSize = 11f
            setTypeface(typeface, Typeface.BOLD)
            val bpad = dp(6)
            setPadding(bpad, dp(2), bpad, dp(2))
        }
        header.addView(liveBadge)

        val textView = TextView(this).apply {
            setTextColor(Color.parseColor("#E0E0EE"))
            textSize = 16f
            setLineSpacing(dp(4).toFloat(), 1.0f)
            setTextIsSelectable(true)
            setText("Listening...")
        }

        card.addView(header)
        card.addView(textView)

        // Insert at top of results container
        resultsContainer.addView(card, 0)
        tvEmptyResults.visibility = View.GONE

        streamingResultCard = card
        streamingResultTextView = textView
    }

    /** Update the streaming card text in real-time */
    private fun updateStreamingResult(text: String) {
        ensureStreamingCard()
        streamingResultTextView?.text = text.ifBlank { "Listening..." }
    }

    /** Replace the live streaming card with a permanent result card, then return to idle */
    private fun finalizeStreamingResult(text: String) {
        // Remove live streaming card
        streamingResultCard?.let { resultsContainer.removeView(it) }
        streamingResultCard = null
        streamingResultTextView = null

        if (text.isNotBlank()) {
            addResultCard(text)
            val duration = (System.currentTimeMillis() - recordingStartTime) / 1000f
            appendSystemMessage("Omni OK — %.1fs audio decoded".format(duration))
        } else {
            appendSystemMessage("Omni: empty response (check logs)")
        }
        returnToIdle()
    }

    // ══════════════════════════════════════════════
    //  Streaming: Decode & Handle Result
    // ══════════════════════════════════════════════

    private fun decodeAndHandleStreamingResult() {
        // Phase 2: Incremental streaming decode — non-blocking per-token
        val ok = engine?.startDecode() ?: false
        Log.i(TAG, "Streaming startDecode: $ok, isDecoding=${engine?.isDecoding()}")

        if (ok) {
            var lastPartial = ""
            while (engine?.isDecoding() == true) {
                engine?.decodeStep()
                val partial = engine?.getPartialResult() ?: ""
                if (partial.isNotEmpty() && partial != lastPartial) {
                    lastPartial = partial
                    Log.d(TAG, "Streaming partial: $partial")
                    runOnUiThread { setStatus("Decoding: $partial") }
                }
            }
        }
        val text = engine?.getResultText() ?: ""
        Log.i(TAG, "Streaming final: $text, tokens=${engine?.getResult() ?: ""}")
        engine?.reset()

        runOnUiThread {
            if (text.isNotBlank()) {
                addResultCard(text)
            }
            if (stoppedByUser.get()) {
                returnToIdle()
            } else {
                restartStreamingIfNeeded()
            }
        }
    }

    private fun restartStreamingIfNeeded() {
        if (stoppedByUser.get() || currentMode != TestMode.STREAMING) return

        // Small delay before auto-restart
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            if (isDestroyed || isFinishing) return@postDelayed
            if (stoppedByUser.get() || currentMode != TestMode.STREAMING) return@postDelayed

            isRecording.set(true)
            silenceChunkCount = 0
            speechDetected = false
            currentRms = 0f

            btnRecord.text = "STOP"
            btnRecord.setBackgroundResource(R.drawable.bg_rec_button_active)
            btnRecord.isEnabled = true
            setStatus("● Listening...")
            tvStatus.startAnimation(blinkAnimation)

            // Reset audio level
            audioLevelFill.layoutParams.width = 0
            audioLevelFill.requestLayout()

            // Start new recording loop
            initAudioRecord()
            audioRecord?.startRecording()
            val chunkSize = (CHUNK_INTERVAL_MS * SAMPLE_RATE / 1000).toInt()
            recordingThread = Thread { recordingLoop(chunkSize) }
            recordingThread?.start()
        }, 300)
    }

    // ══════════════════════════════════════════════
    //  Batch Result
    // ══════════════════════════════════════════════

    private fun onBatchResult(text: String) {
        if (text.isNotBlank()) {
            addResultCard(text)
        } else {
            appendSystemMessage("→ (no speech detected)")
        }
        returnToIdle()
    }

    // ══════════════════════════════════════════════
    //  UI State Management
    // ══════════════════════════════════════════════

    private fun returnToIdle() {
        // Phase 2: Clean up streaming resources
        stopStreamingTimer()
        streamingIncrementalInProgress = false

        btnRecord.text = "REC"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
        btnRecord.isEnabled = true
        audioLevelContainer.visibility = View.GONE
        tvStatus.clearAnimation()

        when (currentMode) {
            TestMode.BATCH -> setStatus("Ready — tap REC to try again")
            TestMode.STREAMING -> setStatus("Streaming stopped — tap REC to restart")
            TestMode.OMNI -> setStatus("Omni ready — tap REC to test (streaming mode)")
        }
    }

    private fun setStatus(text: String) {
        tvStatus.text = text
        Log.i(TAG, "Status: $text")
    }

    private fun updateAudioLevel(rms: Float) {
        if (!isRecording.get()) return
        audioLevelContainer.visibility = View.VISIBLE
        val ratio = (rms / 5000f).coerceIn(0f, 1f)
        val containerWidth = audioLevelContainer.width
        if (containerWidth > 0) {
            val params = audioLevelFill.layoutParams
            params.width = (containerWidth * ratio).toInt()
            audioLevelFill.layoutParams = params
        }
    }

    // ══════════════════════════════════════════════
    //  Result Cards
    // ══════════════════════════════════════════════

    private fun addResultCard(text: String) {
        tvEmptyResults.visibility = View.GONE
        btnClear.visibility = View.VISIBLE
        resultCardCount++

        val timestamp = timeFormatter.format(Date())

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_asr_result_card)
            val pad = dp(16)
            setPadding(pad, pad, pad, pad)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(12) }
        }

        // Header row: index badge + timestamp
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(8) }
        }

        val indexBadge = TextView(this).apply {
            setText("#$resultCardCount")
            setTextColor(Color.parseColor("#2E65C2"))
            textSize = 12f
            setTypeface(typeface, Typeface.BOLD)
            val bpad = dp(6)
            setPadding(bpad, dp(2), bpad, dp(2))
            setBackgroundResource(R.drawable.bg_mode_chip_normal)
        }
        val timeView = TextView(this).apply {
            setText(timestamp)
            setTextColor(Color.parseColor("#666680"))
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { leftMargin = dp(12); gravity = Gravity.CENTER_VERTICAL }
        }
        header.addView(indexBadge)
        header.addView(timeView)

        // Body
        val textView = TextView(this).apply {
            setText(text)
            setTextColor(Color.parseColor("#E0E0EE"))
            textSize = 16f
            setLineSpacing(dp(4).toFloat(), 1.0f)
            setTextIsSelectable(true)
        }

        card.addView(header)
        card.addView(textView)

        resultsContainer.addView(card, 0) // newest at top

        // Only auto-scroll in batch mode; in streaming mode let the user read freely
        if (currentMode == TestMode.BATCH) {
            findViewById<ScrollView>(R.id.scrollView).post {
                findViewById<ScrollView>(R.id.scrollView).fullScroll(View.FOCUS_UP)
            }
        }
    }

    private fun appendSystemMessage(msg: String) {
        resultsContainer.addView(TextView(this).apply {
            text = msg
            setTextColor(Color.parseColor("#8E8E9E"))
            textSize = 13f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(4); bottomMargin = dp(4) }
        })
        tvEmptyResults.visibility = View.GONE
    }

    private fun clearResults() {
        resultsContainer.removeAllViews()
        resultsContainer.addView(tvEmptyResults)
        tvEmptyResults.visibility = View.VISIBLE
        btnClear.visibility = View.GONE
        resultCardCount = 0
        streamingResultCard = null
        streamingResultTextView = null
    }

    // ══════════════════════════════════════════════
    //  Utility
    // ══════════════════════════════════════════════

    private fun initAudioRecord() {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf * 2
        )
        try {
            if (AcousticEchoCanceler.isAvailable()) {
                aec = AcousticEchoCanceler.create(audioRecord!!.audioSessionId)
                aec?.enabled = true
            }
        } catch (_: Exception) {}
        try {
            if (NoiseSuppressor.isAvailable()) {
                noiseSuppressor = NoiseSuppressor.create(audioRecord!!.audioSessionId)
                noiseSuppressor?.enabled = true
            }
        } catch (_: Exception) {}
    }

    private fun stopAudioHardware() {
        try { aec?.release() } catch (_: Exception) {}
        aec = null
        try { noiseSuppressor?.release() } catch (_: Exception) {}
        noiseSuppressor = null
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {}
        audioRecord = null
    }

    private fun dp(value: Int): Int = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), resources.displayMetrics
    ).toInt()

    override fun onStop() {
        super.onStop()
        tvStatus.clearAnimation()
        stopStreamingTimer()
        // Pause recording when app goes to background
        if (isRecording.get()) {
            stoppedByUser.set(true)
            isRecording.set(false)
            stopAudioHardware()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stoppedByUser.set(true)
        isRecording.set(false)
        tvStatus.clearAnimation()
        stopStreamingTimer()
        stopAudioHardware()
        engine?.release()
        engine = null
        try { llmSession?.release() } catch (_: Exception) {}
        llmSession = null
    }
}
