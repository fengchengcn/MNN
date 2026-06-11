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
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import com.k2fsa.sherpa.mnn.Vad
import com.k2fsa.sherpa.mnn.getVadModelConfig
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

class Qwen3AsrTestActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "Qwen3AsrTest"
        private const val SAMPLE_RATE = 16000
        private const val REQUEST_AUDIO = 100
        private const val MAX_TOTAL_CHUNKS = 1800  // 3 min safety net (1800 × 100ms)
        private const val CHUNK_INTERVAL_MS = 100
    }

    // ── UI ──
    private lateinit var chipToggle: TextView
    private lateinit var btnRecord: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvEmptyResults: TextView
    private lateinit var btnClear: TextView
    private lateinit var audioLevelFill: View
    private lateinit var audioLevelContainer: LinearLayout
    private lateinit var resultsContainer: LinearLayout

    // ── State ──
    private var llmSession: LlmSession? = null
    private var omniModelDir: String? = null
    private var modelLoaded = false
    private var audioRecord: AudioRecord? = null
    private var aec: AcousticEchoCanceler? = null
    private var noiseSuppressor: NoiseSuppressor? = null
    private val isRecording = AtomicBoolean(false)
    private val stoppedByUser = AtomicBoolean(false)
    private var recordingThread: Thread? = null
    private var resultCardCount = 0
    private val timeFormatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    // ── Omni state ──
    @Volatile private var speechDetected = false
    @Volatile private var currentRms = 0f
    private val omniAudioBuffer = mutableListOf<Float>()  // BATCH mode: accumulate PCM float samples

    // ── Silero VAD (neural voice activity detection) ──
    private var vad: Vad? = null
    private val vadWindowSize = 512  // 32ms at 16kHz

    // ── Omni VAD mode segment state ──
    private var omniSegmentCount = 0
    private var omniAllResults = StringBuilder()
    private var idleReturned = false
    private var segmentStartTime = 0L
    private var omniLiveCardId = -1

    // ── Serial segment processing ──
    private data class SegmentTask(val samples: FloatArray, val isFinal: Boolean)
    private lateinit var segmentChannel: Channel<SegmentTask>
    private var segmentConsumerJob: Job? = null

    // ── Omni sub-mode: VAD vs BATCH ──
    private var omniUseVad = true

    // ── Blink animation ──
    private var blinkAnimation: AlphaAnimation? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_qwen3_asr_test)

        // Bind views
        chipToggle = findViewById(R.id.chipOmni)
        btnRecord = findViewById(R.id.btnRecord)
        tvStatus = findViewById(R.id.tvStatus)
        tvEmptyResults = findViewById(R.id.tvEmptyResults)
        btnClear = findViewById(R.id.btnClear)
        audioLevelFill = findViewById(R.id.audioLevelFill)
        audioLevelContainer = findViewById(R.id.audioLevelContainer)
        resultsContainer = findViewById(R.id.resultsContainer)

        // Hide legacy mode chips (no longer used)
        findViewById<TextView>(R.id.chipBatch).visibility = View.GONE
        findViewById<TextView>(R.id.chipStreaming).visibility = View.GONE
        findViewById<LinearLayout>(R.id.engineSelector).visibility = View.GONE

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
        setStatus("Initializing Omni engine...")
        btnRecord.isEnabled = false
        updateToggleChip()
        lifecycleScope.launch(Dispatchers.IO) { initEngine() }

        // Listeners
        chipToggle.setOnClickListener {
            if (!isRecording.get()) {
                omniUseVad = !omniUseVad
                updateToggleChip()
                val label = if (omniUseVad) "VAD-segmented" else "BATCH (full audio)"
                setStatus("Omni $label — tap REC to test")
            }
        }
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
    //  Omni Model Discovery & Loading
    // ══════════════════════════════════════════════

    /**
     * Scan /data/local/tmp/mnn_models/ for Omni-compatible models.
     * Omni: audio.mnn + config.json with is_audio=true (FP16 preferred over INT8).
     */
    private fun findBestOmniModel(): String? {
        val localDir = File("/data/local/tmp/mnn_models")
        if (!localDir.exists() || !localDir.isDirectory) return null

        data class Candidate(val dir: String, val score: Int)

        val candidates = mutableListOf<Candidate>()

        localDir.listFiles()?.forEach { subdir ->
            if (!subdir.isDirectory) return@forEach
            val audioMnn = File(subdir, "audio.mnn")
            val configJson = File(subdir, "config.json")
            if (audioMnn.exists() && configJson.exists()) {
                try {
                    val config = JSONObject(configJson.readText())
                    if (config.optBoolean("is_audio", false)) {
                        val name = subdir.name.uppercase()
                        val score = when {
                            name.contains("FP16") -> 2
                            name.contains("INT8") -> 1
                            else -> 0
                        }
                        candidates.add(Candidate(subdir.absolutePath, score))
                    }
                } catch (_: Exception) {}
            }
        }

        val best = candidates.maxByOrNull { it.score } ?: return null
        Log.i(TAG, "Found Omni model: ${best.dir} (score=${best.score})")
        return best.dir
    }

    private suspend fun initEngine() {
        try {
            val modelDir = findBestOmniModel()
            if (modelDir == null) {
                withContext(Dispatchers.Main) {
                    setStatus("ERROR: No Omni ASR model found")
                    appendSystemMessage("Place model at /data/local/tmp/mnn_models/<model>/")
                    appendSystemMessage("Required: audio.mnn + config.json with is_audio=true")
                    btnRecord.isEnabled = false
                }
                return
            }

            omniModelDir = modelDir
            val configPath = "$modelDir/config.json"
            val session = ChatService.provide().createLlmSession(
                "omni_test",
                configPath,
                "omni_test_${System.currentTimeMillis()}",
                null,
                true,
                "cpu"
            ) as? LlmSession

            if (session == null) {
                Log.e(TAG, "Omni LlmSession creation returned null")
                withContext(Dispatchers.Main) {
                    setStatus("ERROR: Failed to create LlmSession")
                    appendSystemMessage("Check config.json and model files at $modelDir")
                }
                return
            }

            session.load()
            session.setKeepHistory(false)
            llmSession = session
            modelLoaded = true

            Log.i(TAG, "Omni LlmSession loaded OK: $modelDir")
            withContext(Dispatchers.Main) {
                appendSystemMessage("Loaded: Omni (${File(modelDir).name})")
                val label = if (omniUseVad) "VAD-segmented" else "BATCH (full audio)"
                setStatus("Omni $label — tap REC to test")
                btnRecord.isEnabled = true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Engine init error", e)
            withContext(Dispatchers.Main) { setStatus("ERROR: ${e.message}") }
        }
    }

    private fun updateToggleChip() {
        chipToggle.text = if (omniUseVad) "OMNI-VAD" else "OMNI-BATCH"
        chipToggle.setBackgroundResource(R.drawable.bg_mode_chip_selected)
        chipToggle.setTextColor(Color.WHITE)
    }

    // ══════════════════════════════════════════════
    //  Recording — Entry Points
    // ══════════════════════════════════════════════

    private fun startRecording() {
        if (isRecording.get() || !modelLoaded) return

        omniAudioBuffer.clear()
        idleReturned = false
        isRecording.set(true)
        stoppedByUser.set(false)
        speechDetected = false
        currentRms = 0f

        btnRecord.text = "STOP"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_active)

        omniSegmentCount = 0
        omniAllResults.clear()
        omniLiveCardId = -1

        if (omniUseVad) {
            initTestVad()
            setStatus("● Recording — VAD listening...")
        } else {
            setStatus("● Recording (BATCH) — tap STOP when done")
        }

        // Init audio
        initAudioRecord()
        audioRecord?.startRecording()

        // Start serial segment consumer
        segmentChannel = Channel(Channel.UNLIMITED)
        segmentConsumerJob = lifecycleScope.launch(Dispatchers.IO) {
            runSegmentConsumer()
        }

        val chunkSize = (CHUNK_INTERVAL_MS * SAMPLE_RATE / 1000).toInt()
        recordingThread = Thread { recordingLoop(chunkSize) }
        recordingThread?.start()
    }

    private fun stopRecording() {
        isRecording.set(false)
        tvStatus.clearAnimation()
    }

    // ══════════════════════════════════════════════
    //  Recording Loop
    // ══════════════════════════════════════════════

    private fun recordingLoop(chunkSize: Int) {
        val shortBuf = ShortArray(chunkSize)
        var totalChunks = 0

        while (isRecording.get() && audioRecord != null && totalChunks < MAX_TOTAL_CHUNKS) {
            totalChunks++
            val ret = audioRecord?.read(shortBuf, 0, chunkSize) ?: 0
            if (ret <= 0) continue

            val floatBuf = FloatArray(ret) { i -> shortBuf[i] / 32768.0f }

            // Compute RMS for UI
            var sumSq = 0f
            for (i in 0 until ret) {
                val s = shortBuf[i].toFloat()
                sumSq += s * s
            }
            val rms = sqrt(sumSq / ret)
            currentRms = rms
            runOnUiThread { updateAudioLevel(rms) }

            if (!omniUseVad) {
                // ── BATCH mode: accumulate all audio ──
                synchronized(omniAudioBuffer) {
                    omniAudioBuffer.addAll(floatBuf.toList())
                }
            } else {
                // ── Silero VAD-driven segment management ──
                vad?.acceptWaveform(floatBuf)

                if (vad?.isSpeechDetected() == true) {
                    runOnUiThread {
                        setStatus("● Segment #${omniSegmentCount + 1} — speaking...")
                    }
                }

                while (vad?.empty() == false) {
                    val segment = vad!!.front()
                    if (segment.samples.size >= SAMPLE_RATE * 0.15f) {
                        omniSegmentCount++
                        segmentStartTime = System.currentTimeMillis()
                        Log.i(TAG, "VAD segment #$omniSegmentCount: ${segment.samples.size} samples " +
                                "(${"%.1f".format(segment.samples.size / SAMPLE_RATE.toFloat())}s)")

                        runOnUiThread {
                            btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
                            btnRecord.text = "..."
                            btnRecord.isEnabled = false
                            setStatus("Segment #$omniSegmentCount — transcribing...")
                        }
                        segmentChannel.trySend(SegmentTask(segment.samples.copyOf(), isFinal = true))
                    }
                    vad!!.pop()
                }
            }
        }

        if (totalChunks >= MAX_TOTAL_CHUNKS) {
            Log.w(TAG, "Max duration reached")
        }

        // Clean up audio hardware
        stopAudioHardware()
        isRecording.set(false)

        if (!omniUseVad) {
            // ── BATCH mode: send entire recording as one segment ──
            val snapshot = synchronized(omniAudioBuffer) { omniAudioBuffer.toFloatArray() }
            omniAudioBuffer.clear()
            omniSegmentCount = 1

            if (snapshot.isNotEmpty()) {
                runOnUiThread {
                    btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
                    btnRecord.text = "..."
                    btnRecord.isEnabled = false
                    setStatus("BATCH decoding ${snapshot.size} samples (%.1fs)...".format(snapshot.size / SAMPLE_RATE.toFloat()))
                }
                segmentChannel.trySend(SegmentTask(snapshot, isFinal = true))
            }

            segmentChannel.close()
            try { runBlocking { segmentConsumerJob?.join() } } catch (_: Exception) {}
            runOnUiThread { returnToIdle() }
            return
        }

        // ── VAD mode: flush in-progress speech, then close channel ──
        vad?.flush()
        while (vad?.empty() == false) {
            val segment = vad!!.front()
            if (segment.samples.size >= SAMPLE_RATE * 0.15f) {
                omniSegmentCount++
                Log.i(TAG, "VAD flush segment #$omniSegmentCount: ${segment.samples.size} samples")
                runOnUiThread {
                    btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
                    btnRecord.text = "..."
                    btnRecord.isEnabled = false
                    setStatus("Final segment #$omniSegmentCount...")
                }
                segmentChannel.trySend(SegmentTask(segment.samples.copyOf(), isFinal = true))
            }
            vad!!.pop()
        }
        try { vad?.release() } catch (_: Exception) {}
        vad = null

        segmentChannel.close()
        try { runBlocking { segmentConsumerJob?.join() } } catch (_: Exception) {}
        runOnUiThread { returnToIdle() }
    }

    // ══════════════════════════════════════════════
    //  Silero VAD Initialization
    // ══════════════════════════════════════════════

    private fun initTestVad() {
        try {
            val config = getVadModelConfig(0)!!
            config.sileroVadModelConfig.threshold = 0.5f
            config.sileroVadModelConfig.minSilenceDuration = 0.4f
            config.sileroVadModelConfig.minSpeechDuration = 0.15f
            config.sileroVadModelConfig.maxSpeechDuration = 15.0f
            config.sileroVadModelConfig.windowSize = vadWindowSize
            config.numThreads = 1
            config.provider = "cpu"
            vad = Vad(assetManager = assets, config = config)
            Log.i(TAG, "Silero VAD initialized for Omni test mode")
        } catch (e: Exception) {
            Log.e(TAG, "Silero VAD init failed", e)
            vad = null
        }
    }

    // ══════════════════════════════════════════════
    //  Omni Segment Inference
    // ══════════════════════════════════════════════

    private suspend fun runSegmentConsumer() {
        try {
            for (task in segmentChannel) {
                processSegmentSync(task)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Segment consumer error", e)
        }
    }

    private suspend fun processSegmentSync(task: SegmentTask) {
        if (llmSession == null) {
            Log.w(TAG, "processSegmentSync: LlmSession is null, skipping")
            return
        }

        val samples = task.samples
        val segNum = omniSegmentCount
        val segStart = segmentStartTime
        val audioTag = "<audio>stream</audio>"
        Log.i(TAG, "processSegment #$segNum [FINAL]: $audioTag, ${samples.size} samples " +
                "(${"%.1f".format(samples.size / SAMPLE_RATE.toFloat())}s audio)")

        try {
            llmSession?.setAudioData(samples, SAMPLE_RATE)

            var fullText = ""
            llmSession?.generate(audioTag, mapOf(), object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress != null) fullText += progress
                    return false
                }
            })

            val response = fullText
            val parsed = extractAsrText(response)
            Log.i(TAG, "processSegment #$segNum result: $response")

            withContext(Dispatchers.Main) {
                if (parsed.isNotBlank()) {
                    addResultCard(parsed)
                    omniAllResults.append(parsed).append("\n")
                }
                val duration = (System.currentTimeMillis() - segStart) / 1000f
                val audioDur = samples.size / SAMPLE_RATE.toFloat()
                appendSystemMessage("Segment #$segNum OK — %.1fs audio, %.1fs inference".format(audioDur, duration))

                if (isRecording.get()) {
                    btnRecord.isEnabled = true
                    btnRecord.text = "STOP"
                    btnRecord.setBackgroundResource(R.drawable.bg_rec_button_active)
                    setStatus("Segment #$segNum done — listening...")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "processSegment error", e)
            withContext(Dispatchers.Main) {
                appendSystemMessage("OMNI ERROR: ${e.message}")
            }
        }
    }

    private fun extractAsrText(raw: String): String {
        val tagStart = raw.indexOf("<asr_text>")
        if (tagStart < 0) return raw.trim()
        val contentStart = tagStart + "<asr_text>".length
        val tagEnd = raw.indexOf("</asr_text>", contentStart)
        if (tagEnd < 0) return raw.substring(contentStart).trim()
        return raw.substring(contentStart, tagEnd).trim()
    }

    // ══════════════════════════════════════════════
    //  UI State Management
    // ══════════════════════════════════════════════

    private fun returnToIdle() {
        if (idleReturned) return
        idleReturned = true

        llmSession?.reset()

        try { vad?.release() } catch (_: Exception) {}
        vad = null
        omniLiveCardId = -1

        btnRecord.text = "REC"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
        btnRecord.isEnabled = true
        audioLevelContainer.visibility = View.GONE
        tvStatus.clearAnimation()

        val label = if (omniUseVad) "VAD" else "BATCH"
        val segInfo = if (omniSegmentCount > 0) " (${omniSegmentCount} segments)" else ""
        setStatus("Omni $label — tap REC to test$segInfo")
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
    //  Audio Hardware
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
        recordingThread?.join(500)
        recordingThread = null
        stopAudioHardware()
        try { vad?.release() } catch (_: Exception) {}
        vad = null
        try { llmSession?.release() } catch (_: Exception) {}
        llmSession = null
    }
}
