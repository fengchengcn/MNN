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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

enum class TestMode { BATCH, STREAMING }

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
    private var audioRecord: AudioRecord? = null
    private val isRecording = AtomicBoolean(false)
    private val stoppedByUser = AtomicBoolean(false)
    private var recordingThread: Thread? = null
    private var resultCardCount = 0
    private val timeFormatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    // ── Streaming state (accessed from recording thread + main thread) ──
    @Volatile private var silenceChunkCount = 0
    @Volatile private var speechDetected = false
    @Volatile private var currentRms = 0f

    // ── Blink animation ──
    private var blinkAnimation: AlphaAnimation? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_qwen3_asr_test)

        // Bind views
        chipBatch = findViewById(R.id.chipBatch)
        chipStreaming = findViewById(R.id.chipStreaming)
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

    private suspend fun initEngine() {
        try {
            val paths = listOf(
                "/data/local/tmp/mnn_models/Qwen3-ASR-0.6B",
                "/data/local/tmp/asr_models"
            )
            var modelDir: String? = null
            for (p in paths) {
                if (java.io.File(p, "audio_encoder.mnn").exists()) {
                    modelDir = p
                    break
                }
            }
            if (modelDir == null) {
                withContext(Dispatchers.Main) {
                    setStatus("ERROR: No model found")
                    appendSystemMessage("Place model at /data/local/tmp/mnn_models/Qwen3-ASR-0.6B/")
                }
                return
            }
            Log.i(TAG, "Using model dir: $modelDir")
            withContext(Dispatchers.Main) { setStatus("Loading model...") }

            engine = Qwen3AsrEngine()
            val ok = engine!!.init(modelDir, cacheDir.absolutePath, numThreads = 4)

            withContext(Dispatchers.Main) {
                if (ok) {
                    setStatus("Ready — tap REC to start")
                    btnRecord.isEnabled = true
                } else {
                    setStatus("ERROR: Engine init failed")
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
        when (mode) {
            TestMode.BATCH -> {
                chipBatch.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipBatch.setTextColor(Color.WHITE)
                chipStreaming.setBackgroundResource(R.drawable.bg_mode_chip_normal)
                chipStreaming.setTextColor(Color.parseColor("#888899"))
                btnRecord.text = "REC"
                btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
                audioLevelContainer.visibility = View.GONE
                setStatus("Batch mode — tap REC to start")
            }
            TestMode.STREAMING -> {
                chipStreaming.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipStreaming.setTextColor(Color.WHITE)
                chipBatch.setBackgroundResource(R.drawable.bg_mode_chip_normal)
                chipBatch.setTextColor(Color.parseColor("#888899"))
                btnRecord.text = "REC"
                btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
                audioLevelContainer.visibility = View.GONE
                setStatus("Streaming mode — auto endpoint detection")
            }
        }
    }

    // ══════════════════════════════════════════════
    //  Recording — Entry Points
    // ══════════════════════════════════════════════

    private fun startRecording() {
        if (isRecording.get()) return

        engine?.reset()
        isRecording.set(true)
        stoppedByUser.set(false)
        silenceChunkCount = 0
        speechDetected = false
        currentRms = 0f

        btnRecord.text = "STOP"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_active)

        if (currentMode == TestMode.BATCH) {
            setStatus("● Recording... tap STOP when done")
        } else {
            setStatus("● Listening...")
            tvStatus.startAnimation(blinkAnimation)
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

        // Read loop
        while (isRecording.get() && audioRecord != null && totalChunks < MAX_TOTAL_CHUNKS) {
            totalChunks++
            val ret = audioRecord?.read(shortBuf, 0, chunkSize) ?: 0
            if (ret <= 0) continue

            val floatBuf = FloatArray(ret) { i -> shortBuf[i] / 32768.0f }
            engine?.pushAudio(floatBuf)

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

        if (isStreaming) {
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
        btnRecord.text = "REC"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
        btnRecord.isEnabled = true
        audioLevelContainer.visibility = View.GONE
        tvStatus.clearAnimation()

        if (currentMode == TestMode.BATCH) {
            setStatus("Ready — tap REC to try again")
        } else {
            setStatus("Streaming stopped — tap REC to restart")
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
                AcousticEchoCanceler.create(audioRecord!!.audioSessionId).enabled = true
            }
        } catch (_: Exception) {}
        try {
            if (NoiseSuppressor.isAvailable()) {
                NoiseSuppressor.create(audioRecord!!.audioSessionId).enabled = true
            }
        } catch (_: Exception) {}
    }

    private fun stopAudioHardware() {
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
        stopAudioHardware()
        engine?.release()
        engine = null
    }
}
