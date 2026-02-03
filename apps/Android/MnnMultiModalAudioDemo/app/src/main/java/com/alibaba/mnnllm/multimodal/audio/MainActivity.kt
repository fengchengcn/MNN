package com.alibaba.mnnllm.multimodal.audio

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.multimodal.audio.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import timber.log.Timber

class MainActivity : AppCompatActivity(), DownloadListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var audioHandler: AudioHandler
    private lateinit var ttsManager: TtsManager
    private lateinit var downloadManager: ModelDownloadManager

    private var currentImagePath: String? = null
    private var isModelLoaded = false

    // Model ID for download manager (ModelScope source)
    private val omniModelId = "ModelScope/MNN/Qwen2.5-Omni-3B-MNN"

    // Local model path (where download manager saves files)
    private val modelDir: String by lazy {
        filesDir.absolutePath + "/.mnnmodels/MNN/Qwen2.5-Omni-3B-MNN"
    }

    private val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
                    permissions ->
                val recordGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
                if (!recordGranted) {
                    Toast.makeText(this, "Need Audio Permission", Toast.LENGTH_SHORT).show()
                }
            }

    private val pickMedia =
            registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
                uri?.let {
                    binding.imgPreview.setImageURI(it)
                    // Copy to local file for Native C++ access
                    currentImagePath = copyUriToCache(it)
                    binding.sampleText.text = "Image selected. Hold Release 'Record' to chat."
                }
            }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize Timber for logging
        if (Timber.forest().isEmpty()) {
            Timber.plant(Timber.DebugTree())
        }

        // Initialize ApplicationProvider for download manager
        ApplicationProvider.set(application)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        checkPermissions()

        audioHandler = AudioHandler(this)
        ttsManager = TtsManager(this)
        downloadManager = ModelDownloadManager.getInstance(this)
        downloadManager.addListener(this)

        checkAndInitModel()

        binding.btnSelectImage.setOnClickListener { pickMedia.launch("image/*") }

        binding.btnRecord.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (!isModelLoaded) {
                        Toast.makeText(this, "Model not ready...", Toast.LENGTH_SHORT).show()
                        return@setOnTouchListener true
                    }
                    ttsManager.stop()
                    audioHandler.startRecording()
                    binding.sampleText.text = "Listening..."
                    binding.btnRecord.text = "Release to Send"
                    return@setOnTouchListener true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    val wavPath = audioHandler.stopRecording()
                    binding.btnRecord.text = "Hold to Speak"
                    if (wavPath != null) {
                        processAudio(wavPath)
                    }
                    return@setOnTouchListener true
                }
            }
            false
        }
    }

    private fun checkPermissions() {
        val permissions =
                mutableListOf(
                        Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                )

        if (permissions.any {
                    ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
                }
        ) {
            requestPermissionLauncher.launch(permissions.toTypedArray())
        }
    }

    private fun checkAndInitModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            val modelFile = File(modelDir)
            val isDownloaded =
                    modelFile.exists() &&
                            modelFile.isDirectory &&
                            modelFile.listFiles()?.isNotEmpty() == true

            withContext(Dispatchers.Main) {
                if (isDownloaded) {
                    // Model already downloaded, try to load it
                    binding.sampleText.text = "Loading Qwen2.5-Omni model..."
                    loadModel()
                } else {
                    // Need to download
                    binding.sampleText.text = "Model not found. Starting download..."
                    startModelDownload()
                }
            }
        }
    }

    private fun startModelDownload() {
        Timber.d("Starting download for $omniModelId")
        binding.sampleText.text = "Downloading Qwen2.5-Omni (≈4GB)...\nPlease wait."
        downloadManager.startDownload(omniModelId)
    }

    private fun loadModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            val success = loadModel(modelDir)
            withContext(Dispatchers.Main) {
                isModelLoaded = success
                binding.sampleText.text =
                        if (success) "Omni Model Loaded. Ready!" else "Model Load Failed"
            }
        }
    }

    private fun processAudio(wavPath: String) {
        binding.sampleText.text = "Analysis..."

        lifecycleScope.launch(Dispatchers.IO) {
            // No ASR needed. Omni takes audio directly.

            val prompt = StringBuilder()
            if (currentImagePath != null) {
                prompt.append("<img>$currentImagePath</img>")
            }
            // Add audio tag (MNN format usually supports <audio>path</audio> for multimodal LLM)
            prompt.append("<audio>$wavPath</audio>")

            withContext(Dispatchers.Main) {
                binding.sampleText.text = "User: [Audio Input]\nAI Thinking..."
            }

            // Send directly to chat
            nativeChat(prompt.toString())
        }
    }

    // Called from JNI
    fun onChatStreamUpdate(chunk: String) {
        runOnUiThread {
            if (!binding.sampleText.text.contains("AI:")) {
                binding.sampleText.append("\nAI: ")
            }
            binding.sampleText.append(chunk)
            ttsManager.speak(chunk)
        }
    }

    private fun copyUriToCache(uri: Uri): String? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val file = File(cacheDir, "temp_image.jpg")
            val outputStream: OutputStream = FileOutputStream(file)
            inputStream?.copyTo(outputStream)
            inputStream?.close()
            outputStream.close()
            file.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // ===== DownloadListener Implementation =====

    override fun onDownloadStart(modelId: String) {
        if (modelId == omniModelId) {
            runOnUiThread { binding.sampleText.text = "Download started..." }
        }
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        if (modelId == omniModelId) {
            runOnUiThread {
                val progress = (downloadInfo.progress * 100).toInt()
                val speedInfo = downloadInfo.speedInfo ?: ""
                binding.sampleText.text = "Downloading: $progress%\n$speedInfo"
            }
        }
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        if (modelId == omniModelId) {
            Timber.d("Download finished: $path")
            runOnUiThread {
                binding.sampleText.text = "Download complete! Loading model..."
                loadModel()
            }
        }
    }

    override fun onDownloadFailed(modelId: String, exception: Exception) {
        if (modelId == omniModelId) {
            Timber.e(exception, "Download failed")
            runOnUiThread {
                binding.sampleText.text = "Download failed: ${exception.message}\nTap to retry."
                binding.sampleText.setOnClickListener {
                    binding.sampleText.setOnClickListener(null)
                    startModelDownload()
                }
            }
        }
    }

    override fun onDownloadPaused(modelId: String) {
        if (modelId == omniModelId) {
            runOnUiThread {
                binding.sampleText.text = "Download paused. Tap to resume."
                binding.sampleText.setOnClickListener {
                    binding.sampleText.setOnClickListener(null)
                    startModelDownload()
                }
            }
        }
    }

    override fun onDownloadTotalSize(modelId: String, totalSize: Long) {
        // Optional: can show total size info
    }

    override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {
        // Optional: handle model updates
    }

    override fun onDownloadFileRemoved(modelId: String) {
        // Optional: handle file removal
    }

    override fun onDestroy() {
        super.onDestroy()
        downloadManager.removeListener(this)
        ttsManager.release()
    }

    // JNI declarations
    external fun loadModel(modelDir: String): Boolean
    external fun nativeChat(prompt: String): Boolean

    companion object {
        init {
            System.loadLibrary("mnn_multimodal_audio")
        }
    }
}
