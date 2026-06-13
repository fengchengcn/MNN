package com.alibaba.mnnllm.android.asr

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
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
import com.alibaba.mnnllm.android.llm.AudioPreprocessor
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.CompletableDeferred
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
    private lateinit var paramsHeader: LinearLayout
    private lateinit var paramsArrow: TextView
    private lateinit var paramsCard: LinearLayout
    private var paramsExpanded = false

    // ── State ──
    private var llmSession: LlmSession? = null
    private var omniModelDir: String? = null
    private var modelLoaded = false
    private var audioRecord: AudioRecord? = null
    // AEC/NS removed — matching sherpa-onnx: raw MIC audio, no hardware effects
    private val isRecording = AtomicBoolean(false)
    private var recordingThread: Thread? = null
    private var recordingSessionId = 0
    private var resultCardCount = 0
    private val timeFormatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    // ── Omni state ──
    @Volatile private var currentRms = 0f
    private val omniAudioBuffer = mutableListOf<Float>()  // BATCH mode: accumulate PCM float samples

    // ── Silero VAD (neural voice activity detection) ──
    private var vad: Vad? = null
    private val vadWindowSize = 512  // 32ms at 16kHz
    // WebRTC-style AGC: normalizes mic distance before VAD
    private val audioPreprocessor = AudioPreprocessor()
    // Locked-in gain: computed once from the first second of audio, then
    // applied uniformly to ALL chunks. This preserves intra-utterance
    // dynamics (same gain for every sample in a sentence) while giving
    // VAD a consistent level — same principle as BATCH mode.
    private var lockedGain = 1.0f
    private var gainLocked = false
    private var preGainChunkCount = 0
    private var preGainRmsAccum = 0f

    // ── Omni VAD mode segment state ──
    private var omniSegmentCount = 0
    private var omniAllResults = StringBuilder()
    private var idleReturned = false
    private var segmentStartTime = 0L

    // ── Cumulative sliding-window inference (VAD mode) ──
    // Design:
    //   VAD segments accumulate within a sentence. After each new segment,
    //   the FULL accumulated buffer is sent for inference → progressive
    //   results with auto-correction (later context disambiguates earlier).
    //
    //   When the wall-clock gap between VAD segments exceeds 3s, that's a
    //   sentence boundary. The current sentence is flushed (final inference),
    //   and a fresh accumulation starts for the next sentence.
    //
    //   Using Channel.UNLIMITED (not CONFLATED) so sentence boundaries work:
    //   when flushing a sentence, we drain stale interim tasks, send the
    //   final task, then start fresh for the next sentence.
    private val accumulatedSegments = mutableListOf<FloatArray>()  // segments in current sentence
    private var totalSpeechSamples = 0        // total speech samples in current sentence
    private var cumulativeRound = 0           // inference round within current sentence
    private var sentenceIndex = 0             // which sentence we're on
    private var lastSegmentWallTime = 0L      // System.currentTimeMillis() of last VAD segment
    private val SENTENCE_BOUNDARY_GAP_MS = 1500L  // 1.5s gap → new sentence

    // ── Serial segment processing ──
    private data class SegmentTask(
        val samples: FloatArray,
        val isFinal: Boolean,
        val sentIndex: Int,   // sentenceIndex at send time (captured, not read live)
        val sentRound: Int    // cumulativeRound at send time (captured, not read live)
    )
    private lateinit var segmentChannel: Channel<SegmentTask>
    private var segmentConsumerJob: Job? = null

    // Track how many segments were included in the last inference,
    // so flushCurrentSentence() can skip redundant final tasks.
    private var lastInferenceSegmentCount = 0

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
        paramsHeader = findViewById(R.id.paramsHeader)
        paramsArrow = findViewById(R.id.paramsArrow)
        paramsCard = findViewById(R.id.paramsCard)

        // Collapsible parameter panel
        paramsHeader.setOnClickListener {
            paramsExpanded = !paramsExpanded
            if (paramsExpanded) {
                paramsCard.visibility = View.VISIBLE
                paramsArrow.text = "▼"
            } else {
                paramsCard.visibility = View.GONE
                paramsArrow.text = "▶"
            }
        }

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
            appendSystemMessage("原生库加载成功")
        } catch (e: UnsatisfiedLinkError) {
            setStatus("致命错误: 原生库加载失败")
            appendSystemMessage("System.loadLibrary(\"mnnllmapp\") 失败: ${e.message}")
            btnRecord.isEnabled = false
            return
        }

        // Init engine
        setStatus("正在初始化 Omni 引擎...")
        btnRecord.isEnabled = false
        updateToggleChip()
        lifecycleScope.launch(Dispatchers.IO) { initEngine() }

        // Listeners
        chipToggle.setOnClickListener {
            if (!isRecording.get()) {
                omniUseVad = !omniUseVad
                updateToggleChip()
                val label = if (omniUseVad) "VAD分段" else "整段模式"
                setStatus("Omni $label — 点击录音按钮开始")
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
     * Returns list of (absolutePath, displayName) sorted by display name.
     */
    private fun findOmniModelCandidates(): List<Pair<String, String>> {
        val localDir = File("/data/local/tmp/mnn_models")
        if (!localDir.exists() || !localDir.isDirectory) return emptyList()

        val candidates = mutableListOf<Pair<String, String>>()

        localDir.listFiles()?.forEach { subdir ->
            if (!subdir.isDirectory) return@forEach
            // Support both old (audio.mnn) and new (conv_frontend.mnn) model layouts
            val audioMnn = File(subdir, "conv_frontend.mnn").let {
                if (it.exists()) it else File(subdir, "audio.mnn")
            }
            val configJson = File(subdir, "config.json")
            if (audioMnn.exists() && configJson.exists()) {
                try {
                    val config = JSONObject(configJson.readText())
                    if (config.optBoolean("is_audio", false)) {
                        candidates.add(Pair(subdir.absolutePath, subdir.name))
                    }
                } catch (_: Exception) {}
            }
        }

        candidates.sortBy { it.second }
        Log.i(TAG, "Found ${candidates.size} Omni model(s): ${candidates.map { it.second }}")
        return candidates
    }

    /**
     * Show model selection dialog and await user choice.
     * Auto-selects if only one candidate; shows dialog if multiple.
     */
    private suspend fun pickOmniModel(candidates: List<Pair<String, String>>): String? {
        if (candidates.size == 1) return candidates[0].first
        val deferred = CompletableDeferred<String?>()
        val activity = this
        withContext(Dispatchers.Main) {
            val names = candidates.map { it.second }.toTypedArray()
            com.google.android.material.dialog.MaterialAlertDialogBuilder(activity)
                .setTitle("选择模型精度")
                .setItems(names) { _, which ->
                    deferred.complete(candidates[which].first)
                }
                .setOnCancelListener {
                    deferred.complete(null)
                }
                .show()
        }
        return deferred.await()
    }

    private suspend fun initEngine() {
        try {
            val candidates = findOmniModelCandidates()
            if (candidates.isEmpty()) {
                withContext(Dispatchers.Main) {
                    setStatus("错误: 未找到 Omni ASR 模型")
                    appendSystemMessage("请将模型放置到 /data/local/tmp/mnn_models/<模型目录>/")
                    appendSystemMessage("需要: conv_frontend.mnn (或 audio.mnn) + config.json (is_audio=true)")
                    btnRecord.isEnabled = false
                }
                return
            }

            val modelDir = pickOmniModel(candidates)
            if (modelDir == null) {
                withContext(Dispatchers.Main) {
                    setStatus("已取消模型选择")
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
                    setStatus("错误: 创建 LlmSession 失败")
                    appendSystemMessage("请检查 $modelDir 下的 config.json 和模型文件")
                }
                return
            }

            // Write ASR-optimized custom config BEFORE load() so it gets merged
            writeAsrCustomConfig()

            session.load()
            session.setKeepHistory(false)
            llmSession = session
            modelLoaded = true

            // Load the effective merged config for display (still on IO thread)
            val mergedConfig = ModelConfig.loadMergedConfig(
                configPath,
                ModelConfig.getExtraConfigFile("omni_test")
            )
            val modelDirName = File(modelDir).name

            Log.i(TAG, "Omni LlmSession loaded OK: $modelDir")
            withContext(Dispatchers.Main) {
                appendSystemMessage("已加载: Omni ($modelDirName)")
                if (mergedConfig != null) {
                    populateParams(mergedConfig, modelDirName)
                } else {
                    Log.w(TAG, "Merged config null; skipping param display")
                }
                val label = if (omniUseVad) "VAD分段" else "整段模式"
                setStatus("Omni $label — 点击录音按钮开始")
                btnRecord.isEnabled = true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Engine init error", e)
            withContext(Dispatchers.Main) { setStatus("错误: ${e.message}") }
        }
    }

    private fun updateToggleChip() {
        chipToggle.text = if (omniUseVad) "OMNI-VAD" else "OMNI-BATCH"
        chipToggle.setBackgroundResource(R.drawable.bg_mode_chip_selected)
        chipToggle.setTextColor(Color.WHITE)
    }

    // ══════════════════════════════════════════════
    //  ASR-Optimized Custom Config
    // ══════════════════════════════════════════════

    /**
     * Write a custom_config.json that overrides sampling parameters for ASR accuracy.
     * Called BEFORE session.load() so LlmSession merges it with base config.json.
     * Only sets ASR-relevant fields; all others are null (omitted by Gson, base values preserved).
     */
    private fun writeAsrCustomConfig() {
        try {
            val config = ModelConfig(
                llmModel = null, llmWeight = null, backendType = null,
                threadNum = 4,                       // CPU threads
                precision = null, useMmap = null, memory = null,
                systemPrompt = null, promptCache = null,
                samplerType = null,                  // let engine use default path
                mixedSamplers = null,
                temperature = 0.0f,                  // no randomness
                topP = 1.0f,                         // no nucleus cutoff
                topK = null, minP = null, tfsZ = null, typical = null,
                penalty = 1.0f,                      // no repetition penalty
                nGram = null, nGramFactor = null,
                maxNewTokens = 128,                  // per-segment cap
                assistantPromptTemplate = null, penaltySampler = null,
                jinja = null, visualModel = null,
                diffusionMemoryMode = null, diffusionSteps = null,
                imageWidth = null, imageHeight = null, diffusionSeed = null,
                cfgPrompt = null, gridSize = null
            )
            val customConfigPath = ModelConfig.getExtraConfigFile("omni_test")
            ModelConfig.saveConfig(customConfigPath, config)
            Log.i(TAG, "ASR custom config written: $customConfigPath")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write ASR custom config", e)
        }
    }

    // ══════════════════════════════════════════════
    //  Model Parameter Display
    // ══════════════════════════════════════════════

    private fun populateParams(mergedConfig: ModelConfig, modelDirName: String) {
        try {
            paramsCard.removeAllViews()
            paramsCard.visibility = View.GONE
            paramsArrow.text = "▶"
            paramsExpanded = false

            // ── Section: Model Info ──
            addParamSection("模型信息")
            addParamRow("模型", modelDirName)
            addParamRow("后端", mergedConfig.backendType ?: "—")
            addParamRow("精度", mergedConfig.precision ?: "—")
            mergedConfig.threadNum?.let {
                if (it > 0) addParamRow("线程数", it.toString())
            }

            // ── Section: Sampler ──
            addParamSection("采样参数")
            mergedConfig.samplerType?.let {
                if (it.isNotBlank()) addParamRow("采样器", it)
            }
            mergedConfig.mixedSamplers?.let {
                if (it.isNotEmpty()) addParamRow("混合", it.joinToString(", "))
            }
            mergedConfig.temperature?.let { addParamRow("温度", formatValue(it)) }
            mergedConfig.topP?.let { addParamRow("Top-P", formatValue(it)) }
            mergedConfig.topK?.let { addParamRow("Top-K", it.toString()) }
            mergedConfig.minP?.let { addParamRow("Min-P", formatValue(it)) }
            mergedConfig.tfsZ?.let { addParamRow("TFS-Z", formatValue(it)) }
            mergedConfig.typical?.let { addParamRow("Typical", formatValue(it)) }

            // ── Section: Penalty ──
            val hasPenalty = mergedConfig.penalty != null || mergedConfig.penaltySampler != null
            if (hasPenalty) {
                addParamSection("惩罚参数")
                mergedConfig.penalty?.let { addParamRow("重复惩罚", formatValue(it)) }
                mergedConfig.penaltySampler?.let {
                    if (it.isNotBlank()) addParamRow("类型", it)
                }
                mergedConfig.nGram?.let { addParamRow("N-Gram", it.toString()) }
                mergedConfig.nGramFactor?.let { addParamRow("N-Gram 系数", formatValue(it)) }
            }

            // ── Section: Generation ──
            addParamSection("生成参数")
            mergedConfig.maxNewTokens?.let { addParamRow("最大 Token 数", it.toString()) }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to populate params", e)
        }
    }

    private fun addParamSection(title: String) {
        paramsCard.addView(TextView(this).apply {
            text = title.uppercase()
            setTextColor(Color.parseColor("#5E5E72"))
            textSize = 10f
            setTypeface(typeface, Typeface.BOLD)
            letterSpacing = 0.08f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = if (paramsCard.childCount > 0) dp(12) else 0
                bottomMargin = dp(6)
            }
        })
    }

    private fun addParamRow(label: String, value: String) {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(2) }
        }

        row.addView(TextView(this).apply {
            text = label
            setTextColor(Color.parseColor("#888899"))
            textSize = 13f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { width = dp(120) }
        })

        row.addView(TextView(this).apply {
            text = value
            setTextColor(Color.parseColor("#D0D0E0"))
            textSize = 13f
            setTypeface(typeface, Typeface.NORMAL)
        })

        paramsCard.addView(row)
    }

    private fun formatValue(value: Any): String {
        return when (value) {
            is Float -> {
                val d = value.toDouble()
                if (d == d.toLong().toDouble()) d.toLong().toString()
                else String.format("%.4f", d).trimEnd('0').trimEnd('.')
            }
            else -> value.toString()
        }
    }

    // ══════════════════════════════════════════════

    // ══════════════════════════════════════════════
    //  Recording — Entry Points
    // ══════════════════════════════════════════════

    private fun startRecording() {
        if (!modelLoaded) return

        // If already recording, stop first — this is a stop-then-start toggle.
        // Also ensures any previous recording's cleanup is fully done before
        // we create new channel / thread / VAD instances.
        if (isRecording.get()) {
            isRecording.set(false)
            recordingThread?.join(1000)
            segmentConsumerJob?.cancel()
        } else {
            // Ensure previous recording is fully cleaned up.
            // The old recordingThread may still be in VAD cleanup (flush, channel.close, etc.)
            // even though isRecording is already false. Wait for it to finish.
            recordingThread?.join(1000)
            segmentConsumerJob?.cancel()
        }

        // Bump session ID so any stale cleanup from old thread is ignored
        recordingSessionId++

        omniAudioBuffer.clear()
        idleReturned = false
        isRecording.set(true)
        currentRms = 0f

        btnRecord.text = "停止"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_active)

        omniSegmentCount = 0
        omniAllResults.clear()
        // Reset cumulative sliding-window state
        synchronized(accumulatedSegments) { accumulatedSegments.clear() }
        totalSpeechSamples = 0
        cumulativeRound = 0
        sentenceIndex = 0
        lastSegmentWallTime = 0L
        lastInferenceSegmentCount = 0
        lockedGain = 1.0f
        gainLocked = false
        preGainChunkCount = 0
        preGainRmsAccum = 0f

        if (omniUseVad) {
            initTestVad()
            setStatus("● 录音中 — VAD 监听中...")
        } else {
            setStatus("● 录音中 (整段模式) — 完成后点击停止")
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
        val mySessionId = recordingSessionId  // capture — stale cleanup must be a no-op
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
                //
                // LOCKED-GAIN approach (same principle as BATCH mode):
                //  1. Collect the first ~1s of audio to estimate ambient level.
                //  2. Lock in one constant gain for the entire recording.
                //  3. Apply that SAME gain to every chunk.
                //
                // This preserves intra-utterance dynamics — the energy contour
                // within a sentence stays natural because every sample gets
                // the same multiplier. Per-chunk gain was distorting relative
                // frame energies, which hurts fbank feature extraction.
                // Target RMS = 0.5 (-6dBFS) matches omni.cpp's target so C++
                // normalization is a no-op — only one gain stage for both modes.
                val chunkRms = sqrt(sumSq / ret) / 32768.0f
                // ── Running gain estimate: apply from the very FIRST chunk ──
                // Using a running average prevents the "blind first second" bug where
                // speech at the recording start goes to VAD unamplified. The gain
                // converges over ~1s, then locks to preserve intra-utterance dynamics.
                preGainChunkCount++
                preGainRmsAccum += chunkRms
                val avgRms = preGainRmsAccum / preGainChunkCount
                lockedGain = if (avgRms > 0.0008f) {
                    (0.5f / avgRms).coerceIn(1f, 10f)  // target -6dBFS, matches omni.cpp
                } else 1f
                if (preGainChunkCount == 10 && !gainLocked) {
                    gainLocked = true
                    Log.i(TAG, "🔒 GAIN LOCKED: avgRMS=%.6f → gain=%.2fx (target=-6dBFS)".format(avgRms, lockedGain))
                }
                if (!gainLocked) {
                    Log.d(TAG, "PRE-GAIN chunk #$preGainChunkCount: chunkRMS=%.4f avgRMS=%.4f gain=%.2fx".format(
                        chunkRms, avgRms, lockedGain))
                }
                // Apply gain immediately — even during pre-gain window
                if (lockedGain > 1.05f) {
                    if (!gainLocked || preGainChunkCount <= 13) {
                        var rmsBefore = 0f
                        for (i in 0 until ret) {
                            val s = floatBuf[i]
                            rmsBefore += s * s
                        }
                        rmsBefore = sqrt(rmsBefore / ret)
                        for (i in 0 until ret) floatBuf[i] *= lockedGain
                        var rmsAfter = 0f
                        for (i in 0 until ret) rmsAfter += floatBuf[i] * floatBuf[i]
                        rmsAfter = sqrt(rmsAfter / ret)
                        Log.i(TAG, "🔊 GAIN chunk #$preGainChunkCount: rms %.4f → %.4f (gain=%.2fx)".format(
                            rmsBefore, rmsAfter, lockedGain))
                    } else {
                        for (i in 0 until ret) floatBuf[i] *= lockedGain
                    }
                }

                vad?.acceptWaveform(floatBuf)

                if (vad?.isSpeechDetected() == true) {
                    runOnUiThread {
                        setStatus("● 第 ${omniSegmentCount + 1} 段 — 说话中...")
                    }
                }

                while (vad?.empty() == false) {
                    val segment = vad!!.front()
                    if (segment.samples.size >= SAMPLE_RATE * 0.15f) {
                        omniSegmentCount++
                        val segDur = segment.samples.size / SAMPLE_RATE.toFloat()
                        val now = System.currentTimeMillis()
                        val gap = if (lastSegmentWallTime > 0) now - lastSegmentWallTime else 0L
                        lastSegmentWallTime = now

                        // ── Sentence boundary detection ──
                        // If >=3s since the last speech segment, this is a new sentence.
                        // Flush the current sentence (final inference) and start fresh.
                        if (gap >= SENTENCE_BOUNDARY_GAP_MS && accumulatedSegments.isNotEmpty()) {
                            Log.i(TAG, "⏸ Sentence boundary: ${"%.1f".format(gap / 1000f)}s gap → flush sentence #$sentenceIndex")
                            flushCurrentSentence()
                        }

                        // Accumulate into the current sentence buffer
                        synchronized(accumulatedSegments) {
                            accumulatedSegments.add(segment.samples.copyOf())
                            totalSpeechSamples += segment.samples.size
                        }

                        val speechTotal = totalSpeechSamples / SAMPLE_RATE.toFloat()
                        Log.i(TAG, "VAD segment #$omniSegmentCount: ${"%.1f".format(segDur)}s → sentence #$sentenceIndex (${"%.1f".format(speechTotal)}s total)")
                        runOnUiThread {
                            setStatus("句#$sentenceIndex 第 $omniSegmentCount 段 — 累积中 (${"%.1f".format(speechTotal)}s)")
                        }

                        // Send cumulative audio (all segments in current sentence)
                        sendCumulativeTask(isFinal = false)
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
            if (recordingSessionId != mySessionId) return  // stale session — bail out

            val snapshot = synchronized(omniAudioBuffer) { omniAudioBuffer.toFloatArray() }
            omniAudioBuffer.clear()
            omniSegmentCount = 1

            Log.i(TAG, "📦 BATCH MODE: ${snapshot.size} samples (%.1fs) — NO gain applied (raw audio)".format(
                snapshot.size / SAMPLE_RATE.toFloat()))

            if (snapshot.isNotEmpty()) {
                runOnUiThread {
                    btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
                    btnRecord.text = "..."
                    btnRecord.isEnabled = false
                    setStatus("整段解码 ${snapshot.size} 样本 (%.1fs)...".format(snapshot.size / SAMPLE_RATE.toFloat()))
                }
                segmentChannel.trySend(SegmentTask(snapshot, isFinal = true, sentIndex = 0, sentRound = 1))
            }

            segmentChannel.close()
            try { runBlocking { segmentConsumerJob?.join() } } catch (_: Exception) {}
            runOnUiThread { returnToIdle(mySessionId) }
            return
        }

        // ── VAD mode: flush in-progress speech, finalize current sentence ──
        if (recordingSessionId != mySessionId) return  // stale session — bail out

        // Capture channel + consumer references locally so we never accidentally
        // close / join the next session's instances (segmentChannel is a var).
        val myChannel = segmentChannel
        val myConsumer = segmentConsumerJob

        Log.i(TAG, "🔒 VAD RECORDING END — lockedGain=%.2fx gainLocked=%b preGainChunks=%d".format(
            lockedGain, gainLocked, preGainChunkCount))
        vad?.flush()
        while (vad?.empty() == false) {
            val segment = vad!!.front()
            if (segment.samples.size >= SAMPLE_RATE * 0.15f) {
                omniSegmentCount++
                Log.i(TAG, "VAD flush segment #$omniSegmentCount: ${segment.samples.size} samples — accumulate")
                // Note: no sentence boundary check here — flush segments belong to the
                // current sentence (the user stopped recording, not a long pause)
                synchronized(accumulatedSegments) {
                    accumulatedSegments.add(segment.samples.copyOf())
                    totalSpeechSamples += segment.samples.size
                }
            }
            vad!!.pop()
        }
        try { vad?.release() } catch (_: Exception) {}
        vad = null
        // AudioPreprocessor reserved for future use (not active in current pipeline)

        // Drain any stale interim tasks, then send final inference for the
        // current sentence (if any speech was accumulated)
        flushCurrentSentence()

        myChannel.close()
        try { runBlocking { myConsumer?.join() } } catch (_: Exception) {}
        runOnUiThread { returnToIdle(mySessionId) }
    }

    // ══════════════════════════════════════════════
    //  Silero VAD Initialization
    // ══════════════════════════════════════════════

    private fun initTestVad() {
        try {
            val config = getVadModelConfig(0)!!
            config.sileroVadModelConfig.threshold = 0.5f
            config.sileroVadModelConfig.minSilenceDuration = 0.25f
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

    // ── Cumulative inference helpers ──

    /** Send the current accumulated audio for inference.
     *  @param isFinal true for sentence-final inference (last result for this sentence) */
    private fun sendCumulativeTask(isFinal: Boolean) {
        val task = buildCumulativeTask(isFinal, sentenceIndex, cumulativeRound + 1)
        if (task != null) {
            cumulativeRound++
            lastInferenceSegmentCount = accumulatedSegments.size
            segmentStartTime = System.currentTimeMillis()
            val dur = task.samples.size / SAMPLE_RATE.toFloat()
            val label = if (isFinal) "最终" else "累积"
            Log.i(TAG, "📤 $label task sent#$sentenceIndex round #$cumulativeRound: ${"%.1f".format(dur)}s → channel")
            runOnUiThread {
                btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
                btnRecord.text = "..."
                btnRecord.isEnabled = false
                setStatus("句#$sentenceIndex ${label}推理 #$cumulativeRound (${"%.1f".format(dur)}s)...")
            }
            segmentChannel.trySend(task)
        }
    }

    /** Called when a sentence boundary is detected (≥1.5s pause) or recording ends.
     *  Sends final inference ONLY if new segments were added since the last interim
     *  task, then clears accumulation for the next sentence.
     *
     *  NOTE: we do NOT drain the channel. Draining can accidentally remove tasks
     *  from the next sentence (sent after sentenceIndex was incremented on a
     *  previous flush). The consumer processes tasks in order; if a final task
     *  skips (no new audio), the last interim result is already the complete one. */
    private fun flushCurrentSentence() {
        if (accumulatedSegments.isEmpty()) return

        val segCount = accumulatedSegments.size
        // Only send final if new segments arrived since last interim inference.
        // Otherwise the last interim already has the complete result.
        if (segCount > lastInferenceSegmentCount) {
            sendCumulativeTask(isFinal = true)
        } else {
            Log.i(TAG, "  ⏭ Sentence #$sentenceIndex: skipped redundant final (no new segments since last inference)")
        }

        Log.i(TAG, "  ✅ Sentence #$sentenceIndex flushed (${"%.1f".format(totalSpeechSamples / SAMPLE_RATE.toFloat())}s speech, $cumulativeRound rounds)")

        // Start fresh for the next sentence
        synchronized(accumulatedSegments) { accumulatedSegments.clear() }
        totalSpeechSamples = 0
        cumulativeRound = 0
        lastInferenceSegmentCount = 0
        sentenceIndex++
    }

    /** Concatenate all accumulated VAD segments directly into one FloatArray.
     *  No synthetic silence gaps are inserted — zero-padding creates unnatural
     *  constant -1.0 fbank features that the conformer encoder has never seen
     *  during training, which degrades recognition accuracy. */
    private fun buildCumulativeTask(isFinal: Boolean, sentIndex: Int, sentRound: Int): SegmentTask? {
        val segments = synchronized(accumulatedSegments) { accumulatedSegments.toList() }
        if (segments.isEmpty()) return null

        val speechTotal = segments.sumOf { it.size }
        val concatenated = FloatArray(speechTotal)
        var offset = 0
        for (seg in segments) {
            System.arraycopy(seg, 0, concatenated, offset, seg.size)
            offset += seg.size
        }
        return SegmentTask(concatenated, isFinal, sentIndex, sentRound)
    }

    // ── Segment consumer ──

    private suspend fun processSegmentSync(task: SegmentTask) {
        if (llmSession == null) {
            Log.w(TAG, "processSegmentSync: LlmSession is null, skipping")
            return
        }

        val samples = task.samples
        val sent = task.sentIndex   // captured at send time — not affected by later flushes
        val round = task.sentRound  // captured at send time
        val segStart = segmentStartTime
        val audioTag = "<audio>stream</audio>"
        val audioDur = samples.size / SAMPLE_RATE.toFloat()
        val isFinal = task.isFinal
        Log.i(TAG, "processSegment sent#$sent round #$round [${if (isFinal) "FINAL" else "INTERIM"}]: " +
                "${"%.1f".format(audioDur)}s cumulative audio")

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
            Log.i(TAG, "processSegment sent#$sent round #$round result: $response")

            withContext(Dispatchers.Main) {
                if (parsed.isNotBlank()) {
                    addResultCard(parsed)
                    omniAllResults.append(parsed).append("\n")
                }
                val duration = (System.currentTimeMillis() - segStart) / 1000f
                if (isFinal) {
                    appendSystemMessage("✅ 句#$sent 完成 — ${"%.1f".format(audioDur)}s (${"%.1f".format(duration)}s 推理)")
                } else {
                    appendSystemMessage("📝 句#$sent 累积 ${"%.1f".format(audioDur)}s → #$round (${"%.1f".format(duration)}s)")
                }

                if (isRecording.get()) {
                    btnRecord.isEnabled = true
                    btnRecord.text = "停止"
                    btnRecord.setBackgroundResource(R.drawable.bg_rec_button_active)
                    if (isFinal) {
                        setStatus("句#$sent 完成 — 监听中...")
                    } else {
                        setStatus("句#$sent 累积推理 #$round 完成 — 继续监听...")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "processSegment error", e)
            withContext(Dispatchers.Main) {
                appendSystemMessage("OMNI 错误: ${e.message}")
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

    private fun returnToIdle(sessionId: Int) {
        // Guard against stale callbacks from a previous recording session.
        // When the user quickly stops and starts, the old recordingThread may
        // still call returnToIdle() after the new session has already started.
        // Without this check the button would revert to idle mid-recording.
        if (idleReturned || recordingSessionId != sessionId) return
        idleReturned = true

        llmSession?.reset()

        try { vad?.release() } catch (_: Exception) {}
        vad = null
        // AudioPreprocessor reserved for future use (not active in current pipeline)

        btnRecord.text = "录音"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
        btnRecord.isEnabled = true
        audioLevelContainer.visibility = View.GONE
        tvStatus.clearAnimation()

        val label = if (omniUseVad) "VAD" else "整段"
        val segInfo = if (omniSegmentCount > 0) " (${omniSegmentCount} 段)" else ""
        setStatus("Omni $label — 点击录音按钮开始$segInfo")
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
        omniAllResults.clear()

        // If recording in VAD mode, also reset the ongoing speech accumulation
        // so the user gets a clean slate. Without this, old segments continue
        // accumulating and produce stale results right after clearing.
        if (isRecording.get() && omniUseVad) {
            synchronized(accumulatedSegments) { accumulatedSegments.clear() }
            totalSpeechSamples = 0
            cumulativeRound = 0
            lastInferenceSegmentCount = 0
            omniSegmentCount = 0
        }
    }

    // ══════════════════════════════════════════════
    //  Audio Hardware
    // ══════════════════════════════════════════════

    private fun initAudioRecord() {
        // Use MIC (raw) instead of VOICE_COMMUNICATION.
        // VOICE_COMMUNICATION applies hardware AGC + noise suppression + echo
        // cancellation that crush dynamic range and distort speech for ASR.
        // sherpa-onnx uses MIC — raw mic data, no processing. Match that.
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf * 2
        )
        // AEC/NS removed — raw MIC passthrough, matching sherpa-onnx
    }

    private fun stopAudioHardware() {
        // AEC/NS references removed
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
            // Signal the recording loop to stop, then immediately release the
            // mic so we don't keep recording in the background. The recording
            // loop's cleanup path (VAD flush, final inference, channel close)
            // still runs; stopAudioHardware() is null-safe and the loop calls
            // it again as a no-op.
            isRecording.set(false)
            stopAudioHardware()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isRecording.set(false)
        tvStatus.clearAnimation()
        recordingThread?.join(500)
        recordingThread = null
        stopAudioHardware()
        try { vad?.release() } catch (_: Exception) {}
        vad = null
        // AudioPreprocessor reserved for future use (not active in current pipeline)
        try { llmSession?.release() } catch (_: Exception) {}
        llmSession = null
    }
}
