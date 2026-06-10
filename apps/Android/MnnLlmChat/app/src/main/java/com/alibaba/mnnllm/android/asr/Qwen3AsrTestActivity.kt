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

        // ── VAD thresholds for Omni mode (float PCM, normalized to ±1.0) ──
        private const val OMNI_SPEECH_RMS = 0.005f       // speech detection threshold
        private const val OMNI_SILENCE_RMS = 0.003f      // silence threshold
        private const val OMNI_MIN_SPEECH_FRAMES = 5     // 500ms minimum speech before trigger
        private const val OMNI_MAX_SILENCE_FRAMES = 6    // 600ms pause = segment boundary
        private const val OMNI_MAX_SPEECH_DURATION_MS = 8000L  // 8s safety net: force segment end
        private const val OMNI_INCREMENTAL_FIRST_MS = 1500L    // first incremental after speech start
        private const val OMNI_INCREMENTAL_INTERVAL_MS = 3000L // subsequent incremental interval
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

    // ── VAD + Segment state ──
    private enum class VadState { SILENCE, SPEAKING }
    @Volatile private var omniVadState = VadState.SILENCE
    private var omniSilenceFrames = 0
    private var omniSpeechFrames = 0
    private var omniSegmentCount = 0
    private val omniPreSpeechRing = mutableListOf<Float>()  // 1s ring buffer for pre-speech context
    private var omniAllResults = StringBuilder()             // accumulated across segments
    private var idleReturned = false                         // guard against double returnToIdle()
    private var segmentStartTime = 0L                        // for logging segment duration

    // ── Expanding-window incremental state ──
    private var omniIncrementalTimer: java.util.Timer? = null
    private var omniIncrementalCount = 0
    private var omniSpeechStartTime = 0L                     // when current utterance started (ms)
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
        chipOmni.setOnClickListener {
            if (currentMode == TestMode.OMNI && !isRecording.get()) {
                // Toggle VAD / BATCH sub-mode
                omniUseVad = !omniUseVad
                updateOmniChipLabel()
                val label = if (omniUseVad) "VAD-segmented" else "BATCH (full audio)"
                setStatus("Omni $label — tap REC to test")
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
    //  Engine Init
    // ══════════════════════════════════════════════

    /**
     * Scan /data/local/tmp/mnn_models/ for Omni-compatible audio models.
     * Prefers FP16 over INT8; returns the config directory path if found.
     */
    private fun findOmniModel(): String? {
        val localDir = File("/data/local/tmp/mnn_models")
        if (!localDir.exists() || !localDir.isDirectory) return null
        var bestPath: String? = null
        var bestScore = -1  // higher = preferred
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
                        if (score > bestScore) {
                            bestScore = score
                            bestPath = subdir.absolutePath
                        }
                    }
                } catch (_: Exception) {}
            }
        }
        if (bestPath != null) Log.i(TAG, "Found Omni model: $bestPath (score=$bestScore)")
        return bestPath
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
                        "cpu"          // backendType — GPU (OpenCL/Vulkan) both too slow on Mali-G76
                    ) as? LlmSession
                    llmSession?.load()
                    llmSession?.setKeepHistory(false)  // ASR: each call independent, no history accumulation
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
                updateOmniChipLabel()
                val label = if (omniUseVad) "VAD-segmented" else "BATCH (full audio)"
                setStatus("Omni $label — tap REC to test")
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
        omniPreSpeechRing.clear()
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
                    // ── Init VAD + expanding-window state ──
                    omniVadState = VadState.SILENCE
                    omniSilenceFrames = 0
                    omniSpeechFrames = 0
                    omniIncrementalCount = 0
                    omniSpeechStartTime = 0L
                    setStatus("● Recording — waiting for speech...")
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
        // Cancel incremental timer if active — recording loop flush will handle FINAL
        cancelIncrementalTimer()
        // Omni: signal recording thread to stop; any active segment will be finalized
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
                    // ── VAD-driven segment management ──
                    // Compute float-domain RMS for Omni VAD (float PCM, range ~0-1）
                    var floatSumSq = 0f
                    for (i in 0 until ret) {
                        floatSumSq += floatBuf[i] * floatBuf[i]
                    }
                    val floatRms = sqrt(floatSumSq / ret)
                    val isSpeech = floatRms > OMNI_SPEECH_RMS

                    when (omniVadState) {
                        VadState.SILENCE -> {
                            // Keep a 1s ring buffer for pre-speech context
                            synchronized(omniPreSpeechRing) {
                                omniPreSpeechRing.addAll(floatBuf.toList())
                                val ringMax = SAMPLE_RATE  // 1 second
                                while (omniPreSpeechRing.size > ringMax) {
                                    omniPreSpeechRing.removeAt(0)
                                }
                            }
                            if (isSpeech) {
                                omniSpeechFrames++
                                if (omniSpeechFrames >= OMNI_MIN_SPEECH_FRAMES) {
                                    enterSpeakingState()
                                }
                            } else {
                                omniSpeechFrames = 0
                            }
                        }
                        VadState.SPEAKING -> {
                            synchronized(omniAudioBuffer) {
                                omniAudioBuffer.addAll(floatBuf.toList())
                            }
                            if (isSpeech) {
                                omniSilenceFrames = 0
                            } else if (floatRms < OMNI_SILENCE_RMS) {
                                omniSilenceFrames++
                                if (omniSilenceFrames >= OMNI_MAX_SILENCE_FRAMES) {
                                    endCurrentSegment()
                                }
                            }
                            // Max duration safety net: force segment end at 8s
                            if (omniSpeechStartTime > 0 &&
                                System.currentTimeMillis() - omniSpeechStartTime >= OMNI_MAX_SPEECH_DURATION_MS) {
                                Log.i(TAG, "VAD: max speech duration (%.1fs) reached — forcing segment end"
                                    .format((System.currentTimeMillis() - omniSpeechStartTime) / 1000f))
                                endCurrentSegment()
                            }
                        }
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
                omniPreSpeechRing.clear()
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

            // ── VAD mode: Flush + close channel (sherpa-onnx pattern) ──
            if (omniVadState == VadState.SPEAKING) {
                omniVadState = VadState.SILENCE
                omniSilenceFrames = 0
                omniSpeechFrames = 0

                cancelIncrementalTimer()

                val snapshot = synchronized(omniAudioBuffer) { omniAudioBuffer.toFloatArray() }
                omniAudioBuffer.clear()
                omniPreSpeechRing.clear()

                if (snapshot.isNotEmpty()) {
                    runOnUiThread {
                        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
                        btnRecord.text = "..."
                        btnRecord.isEnabled = false
                        setStatus("Final decoding...")
                    }
                    segmentChannel.trySend(SegmentTask(snapshot, isFinal = true))
                }
            }

            // Close channel and wait for consumer to finish all segments
            omniPreSpeechRing.clear()
            omniAudioBuffer.clear()
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

    /** Called when VAD detects sustained speech: enter SPEAKING, start expanding window. */
    private fun enterSpeakingState() {
        if (!isRecording.get()) return  // Race guard: recording stopped between VAD check and this call
        omniVadState = VadState.SPEAKING
        omniSilenceFrames = 0
        omniSegmentCount++
        omniIncrementalCount = 0
        segmentStartTime = System.currentTimeMillis()
        omniSpeechStartTime = segmentStartTime

        // Build segment buffer: pre-speech ring tail (context) + clear for new audio
        synchronized(omniPreSpeechRing) {
            val context = omniPreSpeechRing.toList()
            omniPreSpeechRing.clear()
            synchronized(omniAudioBuffer) {
                omniAudioBuffer.clear()
                omniAudioBuffer.addAll(context)
            }
        }

        // Schedule first incremental inference at 1.5s
        scheduleNextIncremental()

        runOnUiThread {
            setStatus("● Segment #$omniSegmentCount — speaking...")
        }
        Log.i(TAG, "VAD: enterSpeakingState — segment #$omniSegmentCount, " +
                "context=${omniAudioBuffer.size} samples, first incremental in ${OMNI_INCREMENTAL_FIRST_MS}ms")
    }

    /**
     * Called when VAD detects sustained silence (600ms) or max duration (8s): end current
     * segment with a FINAL inference. The expanding window has already been sending incremental
     * snapshots; this sends the complete audio for the authoritative result.
     */
    private fun endCurrentSegment() {
        if (omniVadState != VadState.SPEAKING) return  // Guard: already ended (e.g. silence + max-duration double-trigger)
        omniVadState = VadState.SILENCE
        omniSilenceFrames = 0
        omniSpeechFrames = 0

        // Cancel incremental timer — FINAL inference supersedes all pending incrementals
        cancelIncrementalTimer()

        val snapshot = synchronized(omniAudioBuffer) { omniAudioBuffer.toFloatArray() }
        omniAudioBuffer.clear()

        val segNum = omniSegmentCount
        val durationMs = System.currentTimeMillis() - omniSpeechStartTime
        Log.i(TAG, "VAD: endCurrentSegment — segment #$segNum, ${snapshot.size} samples, " +
                "duration=${durationMs}ms")

        val minSpeechSamples = OMNI_MIN_SPEECH_FRAMES * (CHUNK_INTERVAL_MS * SAMPLE_RATE / 1000)
        if (snapshot.size < minSpeechSamples) {
            Log.i(TAG, "VAD: segment #$segNum too short (${snapshot.size} < $minSpeechSamples), skipping")
            omniLiveCardId = -1  // discard any LIVE card from partial incremental
            runOnUiThread {
                setStatus("Segment too short — listening...")
            }
            return
        }

        runOnUiThread {
            btnRecord.setBackgroundResource(R.drawable.bg_rec_button_processing)
            btnRecord.text = "..."
            btnRecord.isEnabled = false
            setStatus("Segment #$segNum — finalizing...")
        }
        segmentChannel.trySend(SegmentTask(snapshot, isFinal = true))
    }

    // ══════════════════════════════════════════════
    //  Omni Segment Inference (one-shot per VAD segment)
    // ══════════════════════════════════════════════

    /**
     * Serial consumer coroutine: reads SegmentTasks from the Channel and processes them
     * one at a time. This matches the sherpa-onnx pattern — no concurrent inference.
     * Both incremental (expanding window) and FINAL tasks share the same channel,
     * guaranteeing pending_audio_ is never overwritten by concurrent generate() calls.
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
     * Incremental tasks update the LIVE card; FINAL tasks replace it with a permanent
     * result card. The expanding window means every task receives the full accumulated
     * audio [0..current], giving the AE complete bidirectional attention context.
     */
    private suspend fun processSegmentSync(task: SegmentTask) {
        if (llmSession == null) {
            Log.w(TAG, "processSegmentSync: LlmSession is null, skipping")
            return
        }

        val samples = task.samples
        val isFinal = task.isFinal
        val segNum = omniSegmentCount
        val segStart = segmentStartTime
        val audioTag = "<audio>stream</audio>"
        val label = if (isFinal) "FINAL" else "incr"
        Log.i(TAG, "processSegment #$segNum [$label]: $audioTag, ${samples.size} samples " +
                "(${"%.1f".format(samples.size / SAMPLE_RATE.toFloat())}s audio)")

        try {
            llmSession?.setAudioData(samples, SAMPLE_RATE)

            var fullText = ""
            llmSession?.generate(audioTag, mapOf(), object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress != null) {
                        fullText += progress
                        if (!isFinal) {
                            // Incremental: stream tokens to LIVE card
                            val dur = (System.currentTimeMillis() - segStart) / 1000f
                            runOnUiThread {
                                updateStreamingResult(fullText)
                                setStatus("Segment #$segNum — transcribing... (%.1fs)".format(dur))
                            }
                        }
                    }
                    return false
                }
            })

            val response = fullText
            Log.i(TAG, "processSegment #$segNum [$label] result: $response")

            withContext(Dispatchers.Main) {
                if (isFinal) {
                    // FINAL: replace LIVE card with permanent result, or add new card
                    if (omniLiveCardId >= 0) {
                        if (response.isNotBlank()) {
                            finalizeStreamingResult(response)
                        } else {
                            // Remove empty LIVE card — model produced no output
                            val card = resultsContainer.findViewWithTag<LinearLayout>("live_card_$omniLiveCardId")
                            card?.let { resultsContainer.removeView(it) }
                            Log.w(TAG, "Segment #$segNum FINAL: empty response, removing LIVE card")
                        }
                        omniLiveCardId = -1
                    } else if (response.isNotBlank()) {
                        addResultCard(response)
                    }
                    if (response.isNotBlank()) {
                        omniAllResults.append(response).append("\n")
                    }
                    val duration = (System.currentTimeMillis() - segStart) / 1000f
                    val audioDur = samples.size / SAMPLE_RATE.toFloat()
                    appendSystemMessage("Segment #$segNum OK — %.1fs audio, %.1fs inference".format(audioDur, duration))
                } else {
                    // Incremental: update LIVE card (created on first call if needed)
                    if (response.isNotBlank()) {
                        ensureStreamingCard()
                        updateStreamingResult(response)
                    }
                }

                if (isRecording.get() && isFinal) {
                    // Segment ended naturally → prepare for next segment
                    btnRecord.isEnabled = true
                    btnRecord.text = "STOP"
                    btnRecord.setBackgroundResource(R.drawable.bg_rec_button_active)
                    setStatus("Segment #$segNum done — listening...")
                }
                // Note: returnToIdle() is called by recordingLoop after channel closes,
                // not here — that ensures all segments are processed before going idle.
            }
        } catch (e: Exception) {
            Log.e(TAG, "processSegment error", e)
            withContext(Dispatchers.Main) {
                appendSystemMessage("OMNI ERROR: ${e.message}")
            }
        }
    }

    // ══════════════════════════════════════════════
    //  Expanding-Window Incremental Inference
    // ══════════════════════════════════════════════

    /**
     * Schedule the next incremental inference. First call uses OMNI_INCREMENTAL_FIRST_MS (1.5s),
     * subsequent calls use OMNI_INCREMENTAL_INTERVAL_MS (3s). Each fire triggers a snapshot of
     * the full accumulated audio buffer (expanding window: [0..current]).
     */
    private fun scheduleNextIncremental() {
        if (!isRecording.get() || omniVadState != VadState.SPEAKING) return

        val elapsed = System.currentTimeMillis() - omniSpeechStartTime
        val delay = if (omniIncrementalCount == 0) {
            maxOf(OMNI_INCREMENTAL_FIRST_MS - elapsed, 50L)
        } else {
            OMNI_INCREMENTAL_INTERVAL_MS
        }

        omniIncrementalTimer?.cancel()
        omniIncrementalTimer = java.util.Timer("asr-incr", true)
        omniIncrementalTimer?.schedule(object : java.util.TimerTask() {
            override fun run() {
                if (!isRecording.get() || omniVadState != VadState.SPEAKING) return
                triggerIncrementalInference()
                scheduleNextIncremental()  // reschedule for next interval
            }
        }, delay)
        Log.d(TAG, "scheduleNextIncremental: count=${omniIncrementalCount}, delay=${delay}ms, " +
                "elapsed=${elapsed}ms")
    }

    private fun cancelIncrementalTimer() {
        omniIncrementalTimer?.cancel()
        omniIncrementalTimer = null
    }

    /**
     * Capture a full snapshot of the accumulated audio buffer and send it to the serial
     * consumer for incremental inference. The expanding window means each call sends ALL
     * audio from speech start to now — the model sees progressively more context.
     */
    private fun triggerIncrementalInference() {
        val snapshot = synchronized(omniAudioBuffer) { omniAudioBuffer.toFloatArray() }
        if (snapshot.size < SAMPLE_RATE / 2) {
            Log.d(TAG, "triggerIncremental: too short (${snapshot.size} samples), skipping")
            scheduleNextIncremental()
            return
        }

        omniIncrementalCount++
        val segNum = omniSegmentCount
        Log.i(TAG, "triggerIncremental #$omniIncrementalCount for segment #$segNum: " +
                "${snapshot.size} samples (${"%.1f".format(snapshot.size / SAMPLE_RATE.toFloat())}s)")

        runOnUiThread {
            setStatus("● Segment #$segNum — incremental #$omniIncrementalCount...")
        }
        segmentChannel.trySend(SegmentTask(snapshot, isFinal = false))
    }

    // ══════════════════════════════════════════════
    //  LIVE Result Card (expanding window UI)
    // ══════════════════════════════════════════════

    /** Create a LIVE result card with red ● LIVE badge. Idempotent — no-op if already shown. */
    private fun ensureStreamingCard() {
        if (omniLiveCardId >= 0) return
        tvEmptyResults.visibility = View.GONE
        btnClear.visibility = View.VISIBLE
        resultCardCount++
        omniLiveCardId = resultCardCount

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
            tag = "live_card_$omniLiveCardId"
        }

        // Header row: index badge + LIVE badge + timestamp
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(8) }
        }

        val indexBadge = TextView(this).apply {
            setText("#$omniLiveCardId")
            setTextColor(Color.parseColor("#2E65C2"))
            textSize = 12f
            setTypeface(typeface, Typeface.BOLD)
            val bpad = dp(6)
            setPadding(bpad, dp(2), bpad, dp(2))
            setBackgroundResource(R.drawable.bg_mode_chip_normal)
        }
        val liveBadge = TextView(this).apply {
            setText("● LIVE")
            setTextColor(Color.parseColor("#FF4444"))
            textSize = 11f
            setTypeface(typeface, Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { leftMargin = dp(8); gravity = Gravity.CENTER_VERTICAL }
        }
        val timeView = TextView(this).apply {
            setText(timestamp)
            setTextColor(Color.parseColor("#666680"))
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f
            ).apply { leftMargin = dp(12); gravity = Gravity.CENTER_VERTICAL }
        }
        header.addView(indexBadge)
        header.addView(liveBadge)
        header.addView(timeView)

        // Body
        val textView = TextView(this).apply {
            setText("...")
            setTextColor(Color.parseColor("#E0E0EE"))
            textSize = 16f
            setLineSpacing(dp(4).toFloat(), 1.0f)
            tag = "live_text_$omniLiveCardId"
        }

        card.addView(header)
        card.addView(textView)

        resultsContainer.addView(card, 0) // newest at top
        Log.d(TAG, "ensureStreamingCard: LIVE card #$omniLiveCardId created")
    }

    /** Update the LIVE card text with the latest expanding-window result. */
    private fun updateStreamingResult(text: String) {
        if (omniLiveCardId < 0) {
            ensureStreamingCard()
        }
        val card = resultsContainer.findViewWithTag<LinearLayout>("live_card_$omniLiveCardId")
        val textView = card?.findViewWithTag<TextView>("live_text_$omniLiveCardId")
        textView?.text = text
    }

    /**
     * Replace the LIVE card with a permanent result card. Removes the LIVE badge,
     * keeps the final text, and resets omniLiveCardId so the next segment creates a new card.
     */
    private fun finalizeStreamingResult(text: String) {
        if (omniLiveCardId < 0) {
            // No LIVE card — add a normal result card
            if (text.isNotBlank()) {
                addResultCard(text)
            }
            return
        }

        val card = resultsContainer.findViewWithTag<LinearLayout>("live_card_$omniLiveCardId") ?: run {
            addResultCard(text)
            omniLiveCardId = -1
            return
        }

        // Remove LIVE badge from header
        val header = card.getChildAt(0) as? LinearLayout
        header?.let {
            // Remove children with LIVE badge text
            for (i in it.childCount - 1 downTo 0) {
                val child = it.getChildAt(i)
                if (child is TextView && (child.text as? String)?.contains("LIVE") == true) {
                    it.removeViewAt(i)
                    break
                }
            }
        }

        // Update body text and remove LIVE tag
        val textView = card.findViewWithTag<TextView>("live_text_$omniLiveCardId")
        textView?.text = text

        // Clear the LIVE card tag so it's no longer targeted by updateStreamingResult
        card.tag = null
        omniLiveCardId = -1
        Log.d(TAG, "finalizeStreamingResult: LIVE card finalized")
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

        // Cancel any pending incremental timer
        cancelIncrementalTimer()

        // Clear Omni session history between recording sessions.
        // Without this, prompt accumulates across sessions (reset() inside submitNative
        // only clears per-call history, not session-level cached prompt text).
        llmSession?.reset()

        // Clean up VAD state
        omniVadState = VadState.SILENCE
        omniSilenceFrames = 0
        omniSpeechFrames = 0
        omniIncrementalCount = 0
        omniSpeechStartTime = 0L
        omniPreSpeechRing.clear()
        omniLiveCardId = -1

        btnRecord.text = "REC"
        btnRecord.setBackgroundResource(R.drawable.bg_rec_button_idle)
        btnRecord.isEnabled = true
        audioLevelContainer.visibility = View.GONE
        tvStatus.clearAnimation()

        when (currentMode) {
            TestMode.BATCH -> setStatus("Ready — tap REC to try again")
            TestMode.STREAMING -> setStatus("Streaming stopped — tap REC to restart")
            TestMode.OMNI -> {
                val label = if (omniUseVad) "VAD" else "BATCH"
                val segInfo = if (omniSegmentCount > 0) " (${omniSegmentCount} segments)" else ""
                setStatus("Omni-$label ready — tap REC to test$segInfo")
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
        cancelIncrementalTimer()
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
        cancelIncrementalTimer()
        stopAudioHardware()
        engine?.release()
        engine = null
        try { llmSession?.release() } catch (_: Exception) {}
        llmSession = null
    }
}
