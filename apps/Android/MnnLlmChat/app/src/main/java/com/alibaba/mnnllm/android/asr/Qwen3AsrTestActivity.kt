package com.alibaba.mnnllm.android.asr

import android.Manifest
import android.content.SharedPreferences
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
import com.k2fsa.sherpa.mnn.Vad
import com.k2fsa.sherpa.mnn.VadModelConfig
import com.k2fsa.sherpa.mnn.SileroVadModelConfig
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
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

enum class TestMode { BATCH, STREAMING, OMNI }

enum class EngineType { OLD_ENGINE, OMNI }

private data class AvailableEngine(
    val type: EngineType,
    val modelDir: String,
    val label: String
)

class Qwen3AsrTestActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "Qwen3AsrTest"
        private const val SAMPLE_RATE = 16000
        private const val REQUEST_AUDIO = 100

        // Silence detection thresholds (reused from VoiceChatPresenter)
        private const val SPEECH_RMS_THRESHOLD = 400.0f
        private const val SILENCE_RMS_THRESHOLD = 100.0f
        private const val MAX_SILENCE_CHUNKS = 15
        private const val MAX_TOTAL_CHUNKS = 1800  // 3 min safety net (1800 × 100ms)
        private const val CHUNK_INTERVAL_MS = 100

        // SharedPreferences keys
        private const val PREFS_NAME = "qwen3_asr_test"
        private const val KEY_ENGINE_TYPE = "engine_type"
        private const val KEY_ENGINE_DIR = "engine_dir"
    }

    // ── UI ──
    private lateinit var chipBatch: TextView
    private lateinit var chipStreaming: TextView
    private lateinit var chipOmni: TextView
    private lateinit var engineSelector: LinearLayout
    private val engineChips = mutableListOf<TextView>()
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
    private var availableEngines = listOf<AvailableEngine>()
    private var selectedEngine: AvailableEngine? = null
    @Volatile private var isSwitchingEngine = false
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
    private val omniAudioBuffer = mutableListOf<Float>()  // Omni BATCH mode: accumulate PCM float samples

    // ── Silero VAD (neural voice activity detection) ──
    private var vad: Vad? = null
    private val vadWindowSize = 512  // 32ms at 16kHz

    // ── Omni VAD mode segment state ──
    private var omniSegmentCount = 0
    private var omniAllResults = StringBuilder()             // accumulated across segments
    private var idleReturned = false                         // guard against double returnToIdle()
    private var segmentStartTime = 0L                        // for logging segment duration
    private var omniLiveCardId = -1                          // LIVE result card currently shown

    // ── Serial segment processing (sherpa-onnx pattern) ──
    private data class SegmentTask(val samples: FloatArray, val isFinal: Boolean)
    private lateinit var segmentChannel: Channel<SegmentTask>
    private var segmentConsumerJob: Job? = null

    // ── Omni sub-mode: VAD vs BATCH ──
    private var omniUseVad = true  // true=VAD-segmented, false=full-audio BATCH

    // ── Blink animation ──
    private var blinkAnimation: AlphaAnimation? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_qwen3_asr_test)

        // Bind views
        chipBatch = findViewById(R.id.chipBatch)
        chipStreaming = findViewById(R.id.chipStreaming)
        chipOmni = findViewById(R.id.chipOmni)
        engineSelector = findViewById(R.id.engineSelector)
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

        // Hide all mode chips initially — avoid flash of wrong UI before engine detection
        chipBatch.visibility = View.GONE
        chipStreaming.visibility = View.GONE
        chipOmni.visibility = View.GONE

        // Init engine
        setStatus("Initializing engine...")
        btnRecord.isEnabled = false
        lifecycleScope.launch(Dispatchers.IO) { initEngine() }

        // Listeners
        chipBatch.setOnClickListener { switchMode(TestMode.BATCH) }
        chipStreaming.setOnClickListener { switchMode(TestMode.STREAMING) }
        chipOmni.setOnClickListener {
            if (currentMode == TestMode.OMNI && !isRecording.get()) {
                // Toggle VAD / BATCH sub-mode
                omniUseVad = !omniUseVad
                updateOmniChipLabel()
                val label = if (omniUseVad) "VAD-segmented" else "BATCH (full audio)"
                val engLabel = when (selectedEngine?.type) {
                    EngineType.OMNI -> "Omni"
                    else -> ""
                }
                setStatus("$engLabel $label — tap REC to test")
            } else {
                switchMode(TestMode.OMNI)
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
    //  Engine Discovery & Selection
    // ══════════════════════════════════════════════

    /**
     * Scan /data/local/tmp/mnn_models/ for all compatible ASR engines.
     * Returns a list of AvailableEngine, sorted by preference (FP16 > INT8 within each type).
     */
    private fun scanAllEngines(): List<AvailableEngine> {
        val result = mutableListOf<AvailableEngine>()
        val localDir = File("/data/local/tmp/mnn_models")
        if (!localDir.exists() || !localDir.isDirectory) return result

        val omniCandidates = mutableListOf<Pair<String, Int>>()  // (dir, score)

        localDir.listFiles()?.forEach { subdir ->
            if (!subdir.isDirectory) return@forEach

            // ── Old-engine: legacy naming ──
            if (File(subdir, "audio_encoder.mnn").exists()) {
                val label = "Old Engine (${subdir.name})"
                result.add(AvailableEngine(EngineType.OLD_ENGINE, subdir.absolutePath, label))
                Log.i(TAG, "Found old-engine model (legacy): ${subdir.absolutePath}")
            }
            // ── Old-engine: new naming (seperate_embed, no is_audio) ──
            else if (File(subdir, "audio.mnn").exists() &&
                File(subdir, "embeddings_bf16.bin").exists() &&
                File(subdir, "tokenizer.txt").exists()) {
                val configFile = File(subdir, "config.json")
                val isOmni = configFile.exists() && try {
                    JSONObject(configFile.readText()).optBoolean("is_audio", false)
                } catch (_: Exception) { false }
                if (!isOmni) {
                    val label = "Old Engine (${subdir.name})"
                    result.add(AvailableEngine(EngineType.OLD_ENGINE, subdir.absolutePath, label))
                    Log.i(TAG, "Found old-engine model (new format): ${subdir.absolutePath}")
                }
            }

            // ── Omni: audio.mnn + config.json with is_audio=true ──
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
                        omniCandidates.add(subdir.absolutePath to score)
                    }
                } catch (_: Exception) {}
            }
        }

        // Add best Omni model (FP16 preferred, then INT8)
        val bestOmni = omniCandidates.maxByOrNull { it.second }
        if (bestOmni != null) {
            val dir = bestOmni.first
            val label = "Omni (${File(dir).name})"
            result.add(AvailableEngine(EngineType.OMNI, dir, label))
            Log.i(TAG, "Found Omni model: $dir (score=${bestOmni.second})")
        }

        // Also scan legacy asr_models path
        val legacyDir = File("/data/local/tmp/asr_models")
        if (legacyDir.exists() && legacyDir.isDirectory) {
            if (File(legacyDir, "audio_encoder.mnn").exists()) {
                result.add(AvailableEngine(EngineType.OLD_ENGINE, legacyDir.absolutePath, "Old Engine (legacy)"))
                Log.i(TAG, "Found old-engine model in legacy path: ${legacyDir.absolutePath}")
            }
        }

        return result
    }

    /**
     * Load a selected engine. Called from IO dispatcher.
     */
    private suspend fun loadEngine(eng: AvailableEngine): Boolean {
        Log.i(TAG, "Loading engine: ${eng.label} (type=${eng.type})")
        withContext(Dispatchers.Main) {
            setStatus("Loading ${eng.label}...")
            btnRecord.isEnabled = false
            isSwitchingEngine = true
        }

        return try {
            when (eng.type) {
                EngineType.OLD_ENGINE -> {
                    val qwen3Eng = Qwen3AsrEngine()
                    val ok = qwen3Eng.init(eng.modelDir, cacheDir.absolutePath, numThreads = 4)
                    if (ok) {
                        this.engine = qwen3Eng
                        Log.i(TAG, "Qwen3AsrEngine initialized OK")
                        withContext(Dispatchers.Main) {
                            appendSystemMessage("Loaded: ${eng.label}")
                        }
                        true
                    } else {
                        Log.e(TAG, "Qwen3AsrEngine.init() returned false")
                        withContext(Dispatchers.Main) {
                            appendSystemMessage("FAILED: ${eng.label}")
                        }
                        false
                    }
                }
                EngineType.OMNI -> {
                    val configPath = "${eng.modelDir}/config.json"
                    val session = ChatService.provide().createLlmSession(
                        "omni_test",
                        configPath,
                        "omni_test_${System.currentTimeMillis()}",
                        null,
                        true,
                        "cpu"
                    ) as? LlmSession
                    session?.load()
                    session?.setKeepHistory(false)
                    if (session != null) {
                        llmSession = session
                        omniModelDir = eng.modelDir
                        Log.i(TAG, "Omni LlmSession loaded OK")
                        withContext(Dispatchers.Main) {
                            appendSystemMessage("Loaded: ${eng.label}")
                        }
                        true
                    } else {
                        Log.e(TAG, "Omni LlmSession creation returned null")
                        withContext(Dispatchers.Main) {
                            appendSystemMessage("FAILED: ${eng.label}")
                        }
                        false
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Engine load failed: ${eng.label}", e)
            withContext(Dispatchers.Main) {
                appendSystemMessage("ERROR: ${e.message}")
            }
            false
        } finally {
            // Always reset switching flag, regardless of success or caller (initEngine/switchToEngine)
            withContext(Dispatchers.Main) {
                isSwitchingEngine = false
            }
        }
    }

    /**
     * Switch to a different engine. Releases current engine, loads new one.
     * Called from UI thread on engine chip tap.
     */
    private fun switchToEngine(target: AvailableEngine) {
        if (isSwitchingEngine || isRecording.get()) return
        if (selectedEngine?.modelDir == target.modelDir && selectedEngine?.type == target.type) {
            return  // already selected
        }

        // Set guard immediately on UI thread to prevent double-tap race.
        // loadEngine() will reset it in its finally block.
        isSwitchingEngine = true

        // Release current engine
        releaseCurrentEngine()
        selectedEngine = null

        lifecycleScope.launch(Dispatchers.IO) {
            val ok = loadEngine(target)
            withContext(Dispatchers.Main) {
                if (ok) {
                    selectedEngine = target
                    saveEnginePreference(target)
                    updateEngineChipSelection()
                    updateModeChipsForEngine(target.type)
                    selectDefaultMode()
                    setStatus("${target.label} — tap REC to test")
                    btnRecord.isEnabled = true
                } else {
                    // Mark chip as failed (dim it, disable click)
                    engineChips.forEach { chip ->
                        if (chip.text == target.label || chip.text.toString().contains(target.label)) {
                            chip.alpha = 0.3f
                        }
                    }
                    setStatus("Failed to load ${target.label}")
                    // If other engines are available and loaded, keep them selectable
                }
            }
        }
    }

    private fun releaseCurrentEngine() {
        engine?.release()
        engine = null
        try { llmSession?.release() } catch (_: Exception) {}
        llmSession = null
        omniModelDir = null
    }

    /**
     * Main engine initialization flow.
     * Scans for available engines and either auto-selects (single engine or saved pref)
     * or shows the selector for user choice.
     */
    private suspend fun initEngine() {
        try {
            // ── Scan for all available engines ──
            availableEngines = scanAllEngines()

            if (availableEngines.isEmpty()) {
                withContext(Dispatchers.Main) {
                    setStatus("ERROR: No ASR model found")
                    appendSystemMessage("Place model at /data/local/tmp/mnn_models/<model>/")
                    appendSystemMessage("Omni: need audio.mnn + config.json with is_audio=true")
                    appendSystemMessage("Old engine (legacy): audio_encoder.mnn + llm_kv_8bit.mnn + embeddings_bf16.bin")
                    appendSystemMessage("Old engine (new):   audio.mnn + llm.mnn + embeddings_bf16.bin + tokenizer.txt")
                    btnRecord.isEnabled = false
                }
                return
            }

            // ── Single engine: auto-select, hide selector ──
            if (availableEngines.size == 1) {
                val only = availableEngines[0]
                val ok = loadEngine(only)
                withContext(Dispatchers.Main) {
                    if (ok) {
                        selectedEngine = only
                        updateModeChipsForEngine(only.type)
                        selectDefaultMode()
                        setStatus("${only.label} — tap REC to test")
                        btnRecord.isEnabled = true
                    } else {
                        setStatus("ERROR: Failed to load ${only.label}")
                        btnRecord.isEnabled = false
                    }
                }
                return
            }

            // ── Multiple engines: show selector ──
            withContext(Dispatchers.Main) { buildEngineSelectorUI() }

            // Try to restore saved preference
            val savedType = restoreEnginePreferenceType()
            val savedDir = restoreEnginePreferenceDir()
            val preferred = if (savedType != null && savedDir != null) {
                availableEngines.find { it.type.name == savedType && it.modelDir == savedDir }
            } else null

            if (preferred != null) {
                val ok = loadEngine(preferred)
                withContext(Dispatchers.Main) {
                    if (ok) {
                        selectedEngine = preferred
                        updateEngineChipSelection()
                        updateModeChipsForEngine(preferred.type)
                        selectDefaultMode()
                        setStatus("${preferred.label} — tap REC to test")
                        btnRecord.isEnabled = true
                    } else {
                        // Saved engine failed — let user pick another
                        setStatus("Previous engine failed. Select another:")
                        btnRecord.isEnabled = false
                    }
                }
            } else {
                // No saved preference — wait for user choice
                withContext(Dispatchers.Main) {
                    setStatus("Select engine to load")
                    btnRecord.isEnabled = false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Engine init error", e)
            withContext(Dispatchers.Main) { setStatus("ERROR: ${e.message}") }
        }
    }

    // ══════════════════════════════════════════════
    //  Engine Selector UI
    // ══════════════════════════════════════════════

    private fun buildEngineSelectorUI() {
        engineSelector.removeAllViews()
        engineChips.clear()

        for (eng in availableEngines) {
            val chip = TextView(this).apply {
                text = eng.label
                gravity = Gravity.CENTER
                textSize = 12f
                setTypeface(typeface, Typeface.BOLD)
                val padH = dp(14)
                val padV = dp(8)
                setPadding(padH, padV, padH, padV)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { marginEnd = dp(4) }
                setTextColor(Color.parseColor("#888899"))
                background = resources.getDrawable(R.drawable.bg_mode_chip_normal, null)
                setOnClickListener { switchToEngine(eng) }
            }
            engineSelector.addView(chip)
            engineChips.add(chip)
        }

        engineSelector.visibility = View.VISIBLE
        Log.i(TAG, "Engine selector built: ${availableEngines.size} engines")
    }

    private fun updateEngineChipSelection() {
        val sel = selectedEngine ?: return
        val selColor = Color.WHITE
        val dimColor = Color.parseColor("#888899")

        for (chip in engineChips) {
            if (chip.text == sel.label || chip.text.toString().contains(sel.label)) {
                chip.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chip.setTextColor(selColor)
                chip.alpha = 1.0f
            } else {
                chip.setBackgroundResource(R.drawable.bg_mode_chip_normal)
                chip.setTextColor(dimColor)
                // Keep failed chips dimmed
                if (chip.alpha > 0.5f) chip.alpha = 1.0f
            }
        }
    }

    /**
     * Show/hide mode chips based on engine capabilities.
     * Old Engine → Batch + Streaming
     * Omni       → Omni-VAD toggle
     */
    private fun updateModeChipsForEngine(type: EngineType) {
        when (type) {
            EngineType.OLD_ENGINE -> {
                chipBatch.visibility = View.VISIBLE
                chipStreaming.visibility = View.VISIBLE
                chipOmni.visibility = View.GONE
            }
            EngineType.OMNI -> {
                chipBatch.visibility = View.GONE
                chipStreaming.visibility = View.GONE
                chipOmni.visibility = View.VISIBLE
                updateOmniChipLabel()
            }
        }
    }

    /** Auto-select a valid default mode for the current engine */
    private fun selectDefaultMode() {
        val type = selectedEngine?.type ?: return
        when (type) {
            EngineType.OLD_ENGINE -> {
                currentMode = TestMode.BATCH
                chipBatch.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipBatch.setTextColor(Color.WHITE)
                chipStreaming.setBackgroundResource(R.drawable.bg_mode_chip_normal)
                chipStreaming.setTextColor(Color.parseColor("#888899"))
            }
            EngineType.OMNI -> {
                currentMode = TestMode.OMNI
                chipOmni.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipOmni.setTextColor(Color.WHITE)
            }
        }
    }

    // ══════════════════════════════════════════════
    //  Preference Persistence
    // ══════════════════════════════════════════════

    private fun getPrefs(): SharedPreferences =
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE)

    private fun saveEnginePreference(eng: AvailableEngine) {
        getPrefs().edit()
            .putString(KEY_ENGINE_TYPE, eng.type.name)
            .putString(KEY_ENGINE_DIR, eng.modelDir)
            .apply()
    }

    private fun restoreEnginePreferenceType(): String? =
        getPrefs().getString(KEY_ENGINE_TYPE, null)

    private fun restoreEnginePreferenceDir(): String? =
        getPrefs().getString(KEY_ENGINE_DIR, null)


    // ══════════════════════════════════════════════
    //  Mode Switching
    // ══════════════════════════════════════════════

    private fun switchMode(mode: TestMode) {
        if (isRecording.get()) return

        // Guard: reject OMNI mode when Old Engine is active (no LlmSession)
        if (mode == TestMode.OMNI && selectedEngine?.type != EngineType.OMNI) {
            Log.w(TAG, "switchMode: OMNI mode rejected — engine type is ${selectedEngine?.type}")
            return
        }

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

        val engLabel = when (selectedEngine?.type) {
            EngineType.OLD_ENGINE -> "Old Engine"
            EngineType.OMNI -> "Omni"
            null -> ""
        }
        when (mode) {
            TestMode.BATCH -> {
                chipBatch.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipBatch.setTextColor(Color.WHITE)
                setStatus("$engLabel Batch — tap REC to start")
            }
            TestMode.STREAMING -> {
                chipStreaming.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipStreaming.setTextColor(Color.WHITE)
                setStatus("$engLabel Streaming — auto endpoint detection")
            }
            TestMode.OMNI -> {
                chipOmni.setBackgroundResource(R.drawable.bg_mode_chip_selected)
                chipOmni.setTextColor(Color.WHITE)
                updateOmniChipLabel()
                val label = if (omniUseVad) "VAD-segmented" else "BATCH (full audio)"
                setStatus("$engLabel $label — tap REC to test")
            }
        }
    }

    private fun updateOmniChipLabel() {
        chipOmni.text = if (omniUseVad) "OMNI-VAD" else "OMNI-BATCH"
    }

    // ══════════════════════════════════════════════
    //  Recording — Entry Points
    // ══════════════════════════════════════════════

    private fun startRecording() {
        if (isRecording.get()) return

        engine?.reset()
        omniAudioBuffer.clear()
        idleReturned = false
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
                omniSegmentCount = 0
                omniAllResults.clear()
                omniLiveCardId = -1
                if (omniUseVad) {
                    // ── Init Silero VAD for neural speech detection ──
                    initTestVad()
                    setStatus("● Recording — VAD listening...")
                } else {
                    // ── BATCH mode: accumulate until STOP ──
                    setStatus("● Recording (BATCH) — tap STOP when done")
                }
            }
        }

        // Init audio
        initAudioRecord()
        audioRecord?.startRecording()

        // Start serial segment consumer (sherpa-onnx pattern: one consumer, no concurrent inference)
        if (currentMode == TestMode.OMNI) {
            segmentChannel = Channel(Channel.UNLIMITED)
            segmentConsumerJob = lifecycleScope.launch(Dispatchers.IO) {
                runSegmentConsumer()
            }
        }

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
        // Omni: signal recording thread to stop; VAD flush will handle final segments
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

            // Compute RMS early — needed by VAD before buffer operations
            var sumSq = 0f
            for (i in 0 until ret) {
                val s = shortBuf[i].toFloat()
                sumSq += s * s
            }
            val rms = sqrt(sumSq / ret)
            currentRms = rms
            runOnUiThread { updateAudioLevel(rms) }

            if (isOmni) {
                if (!omniUseVad) {
                    // ── BATCH mode: just accumulate all audio ──
                    synchronized(omniAudioBuffer) {
                        omniAudioBuffer.addAll(floatBuf.toList())
                    }
                } else {
                    // ── Silero VAD-driven segment management ──
                    // Feed audio to neural VAD model (replaces energy-based RMS state machine)
                    vad?.acceptWaveform(floatBuf)

                    // Update UI based on speech detection
                    if (vad?.isSpeechDetected() == true) {
                        runOnUiThread {
                            setStatus("● Segment #${omniSegmentCount + 1} — speaking...")
                        }
                    }

                    // Extract completed speech segments from VAD
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
            } else {
                engine?.pushAudio(floatBuf)
            }

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

                // Close channel and wait for consumer to finish
                segmentChannel.close()
                try {
                    runBlocking { segmentConsumerJob?.join() }
                } catch (_: Exception) {}
                runOnUiThread { returnToIdle() }
                return
            }

            // ── Silero VAD mode: flush any in-progress speech, then close channel ──
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
            // Release VAD model resources
            try { vad?.release() } catch (_: Exception) {}
            vad = null

            // Close channel and wait for consumer to finish all segments
            segmentChannel.close()
            try {
                runBlocking { segmentConsumerJob?.join() }
            } catch (_: Exception) {}
            runOnUiThread { returnToIdle() }
            return
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
    //  Silero VAD Initialization
    // ══════════════════════════════════════════════

    /** Initialize Silero VAD model from Android assets. Called from startRecording(). */
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
            Log.i(TAG, "Silero VAD initialized for test mode")
        } catch (e: Exception) {
            Log.e(TAG, "Silero VAD init failed", e)
            vad = null
        }
    }

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
    //  VAD State Machine
    // ══════════════════════════════════════════════

    // ══════════════════════════════════════════════
    //  Omni Segment Inference (one-shot per Silero VAD segment)
    // ══════════════════════════════════════════════

    /**
     * Serial consumer coroutine: reads SegmentTasks from the Channel and processes them
     * one at a time. This matches the sherpa-onnx pattern — no concurrent inference.
     * All segments are FINAL (no incremental inference in this simplified version).
     */
    private suspend fun runSegmentConsumer() {
        try {
            for (task in segmentChannel) {
                processSegmentSync(task)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Segment consumer error", e)
        }
    }

    /**
     * Synchronous segment processing — called from the serial consumer coroutine.
     * Each VAD segment is decoded as a complete utterance via the Omni LlmSession.
     */
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
                    if (progress != null) {
                        fullText += progress
                    }
                    return false
                }
            })

            val response = fullText
            Log.i(TAG, "processSegment #$segNum result: $response")

            withContext(Dispatchers.Main) {
                if (response.isNotBlank()) {
                    addResultCard(response)
                    omniAllResults.append(response).append("\n")
                }
                val duration = (System.currentTimeMillis() - segStart) / 1000f
                val audioDur = samples.size / SAMPLE_RATE.toFloat()
                appendSystemMessage("Segment #$segNum OK — %.1fs audio, %.1fs inference".format(audioDur, duration))

                if (isRecording.get()) {
                    // Segment ended naturally → prepare for next segment
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
        if (idleReturned) return
        idleReturned = true

        // Clear Omni session history between recording sessions.
        llmSession?.reset()

        // Clean up VAD state
        omniLiveCardId = -1

        btnRecord.text = "REC"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
        btnRecord.isEnabled = true
        audioLevelContainer.visibility = View.GONE
        tvStatus.clearAnimation()

        val engLabel = when (selectedEngine?.type) {
            EngineType.OLD_ENGINE -> "Old Engine"
            EngineType.OMNI -> "Omni"
            null -> ""
        }
        when (currentMode) {
            TestMode.BATCH -> setStatus("$engLabel Batch — tap REC to try again")
            TestMode.STREAMING -> setStatus("$engLabel Streaming — tap REC to restart")
            TestMode.OMNI -> {
                val label = if (omniUseVad) "VAD" else "BATCH"
                val segInfo = if (omniSegmentCount > 0) " (${omniSegmentCount} segments)" else ""
                setStatus("$engLabel $label — tap REC to test$segInfo")
            }
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

        // Only auto-scroll in batch mode; in VAD/streaming mode let the user read freely
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
        // Join recording thread BEFORE releasing VAD to prevent use-after-free
        recordingThread?.join(500)
        recordingThread = null
        stopAudioHardware()
        try { vad?.release() } catch (_: Exception) {}
        vad = null
        engine?.release()
        engine = null
        try { llmSession?.release() } catch (_: Exception) {}
        llmSession = null
    }
}
