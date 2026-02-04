package com.alibaba.mnnllm.multimodal.audio

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.text.Editable
import android.text.TextWatcher
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.DownloadFileUtils
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

    private lateinit var chatAdapter: ChatAdapter

    private var currentImagePath: String? = null
    private var isModelLoaded = false
    private var loadingModel = false
    private var actualModelPath: String? = null

    // Model ID for download manager
    private val omniModelId = "ModelScope/MNN/Qwen2.5-Omni-3B-MNN"

    // Persistent storage path
    private val PERSISTENT_CACHE_DIR =
            Environment.getExternalStorageDirectory().absolutePath + "/MnnModels"

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
                    val path = copyUriToCache(it)
                    if (path != null) {
                        currentImagePath = path
                        showImagePreview(path)
                    }
                }
            }

    private val requestAllFilesAccessLauncher =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
                if (checkAllFilesAccess()) {
                    setupDownloaderAndStart(PERSISTENT_CACHE_DIR)
                } else {
                    Toast.makeText(
                                    this,
                                    "Permission denied. Models will be stored in app cache.",
                                    Toast.LENGTH_SHORT
                            )
                            .show()
                    setupDownloaderAndStart(filesDir.absolutePath + "/.mnnmodels")
                }
            }

    private fun setupDownloaderAndStart(path: String) {
        ModelDownloadManager.setCacheDir(path)
        ModelDownloadManager.resetInstance()
        downloadManager = ModelDownloadManager.getInstance(this)
        downloadManager.addListener(this)
        checkAndInitModel()
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Timber.forest().isEmpty()) {
            Timber.plant(Timber.DebugTree())
        }

        ApplicationProvider.set(application)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        chatAdapter = ChatAdapter()
        binding.chatRecyclerView.layoutManager =
                LinearLayoutManager(this).apply { stackFromEnd = true }
        binding.chatRecyclerView.adapter = chatAdapter

        audioHandler = AudioHandler(this)
        ttsManager = TtsManager(this)

        if (checkAllFilesAccess()) {
            setupDownloaderAndStart(PERSISTENT_CACHE_DIR)
        } else {
            requestAllFilesAccess()
        }

        setupInputListeners()
    }

    private fun checkAllFilesAccess(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else {
            true
        }
    }

    private fun requestAllFilesAccess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            try {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                intent.data = Uri.parse("package:$packageName")
                requestAllFilesAccessLauncher.launch(intent)
                Toast.makeText(
                                this,
                                "Please grant All Files Access to store models persistently",
                                Toast.LENGTH_LONG
                        )
                        .show()
            } catch (e: Exception) {
                val intent = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                requestAllFilesAccessLauncher.launch(intent)
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupInputListeners() {
        binding.btnAddImage.setOnClickListener {
            checkAndRequestStoragePermission { pickMedia.launch("image/*") }
        }

        binding.btnClosePreview.setOnClickListener { clearImagePreview() }

        binding.inputMessage.addTextChangedListener(
                object : TextWatcher {
                    override fun beforeTextChanged(
                            s: CharSequence?,
                            start: Int,
                            count: Int,
                            after: Int
                    ) {}
                    override fun onTextChanged(
                            s: CharSequence?,
                            start: Int,
                            before: Int,
                            count: Int
                    ) {
                        updateSendButtonState(s.toString())
                    }
                    override fun afterTextChanged(s: Editable?) {}
                }
        )

        binding.btnRecordOrSend.setOnClickListener {
            val text = binding.inputMessage.text.toString().trim()
            if (text.isNotEmpty()) {
                handleSendText(text)
            }
        }

        binding.btnRecordOrSend.setOnTouchListener { _, event ->
            val text = binding.inputMessage.text.toString().trim()
            if (text.isNotEmpty()) return@setOnTouchListener false

            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (!isPermissionGranted(Manifest.permission.RECORD_AUDIO)) {
                        requestPermissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
                        return@setOnTouchListener true
                    }

                    if (!isModelLoaded) {
                        Toast.makeText(this, "Model not ready...", Toast.LENGTH_SHORT).show()
                        return@setOnTouchListener true
                    }
                    ttsManager.stop()
                    audioHandler.startRecording()
                    showRecordingIndicator(true)
                    return@setOnTouchListener true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    showRecordingIndicator(false)
                    val wavPath = audioHandler.stopRecording()
                    if (wavPath != null) {
                        processAudio(wavPath)
                    }
                    return@setOnTouchListener true
                }
            }
            false
        }
    }

    private fun updateSendButtonState(text: String) {
        if (text.trim().isNotEmpty()) {
            binding.btnRecordOrSend.setImageResource(R.drawable.ic_send)
        } else {
            binding.btnRecordOrSend.setImageResource(R.drawable.ic_mic)
        }
    }

    private fun showImagePreview(path: String) {
        binding.previewCard.visibility = View.VISIBLE
        try {
            val bitmap = BitmapFactory.decodeFile(path)
            binding.imgPreviewMini.setImageBitmap(bitmap)
        } catch (e: Exception) {
            Timber.e(e, "Failed to load preview")
        }
    }

    private fun clearImagePreview() {
        currentImagePath = null
        binding.previewCard.visibility = View.GONE
        binding.imgPreviewMini.setImageDrawable(null)
    }

    private fun showRecordingIndicator(show: Boolean) {
        binding.recordingIndicator.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun handleSendText(text: String) {
        if (!isModelLoaded) {
            Toast.makeText(this, "Model is loading...", Toast.LENGTH_SHORT).show()
            return
        }

        val message = ChatMessage(text, true, currentImagePath)
        chatAdapter.addMessage(message)
        scrollToBottom()

        binding.inputMessage.setText("")
        val imagePathToSend = currentImagePath
        clearImagePreview()

        processText(text, imagePathToSend)
    }

    private fun processText(text: String, imagePath: String?) {
        lifecycleScope.launch(Dispatchers.IO) {
            val prompt = StringBuilder()
            if (imagePath != null) {
                prompt.append("<img>$imagePath</img>")
            }
            prompt.append(text)

            withContext(Dispatchers.Main) {
                chatAdapter.addMessage(ChatMessage("Thinking...", false))
                scrollToBottom()
            }

            nativeChat(prompt.toString())
        }
    }

    private fun processAudio(wavPath: String) {
        chatAdapter.addMessage(ChatMessage("[Audio Message]", true, isAudio = true))
        scrollToBottom()

        val imagePathToSend = currentImagePath
        clearImagePreview()

        lifecycleScope.launch(Dispatchers.IO) {
            val prompt = StringBuilder()
            if (imagePathToSend != null) {
                prompt.append("<img>$imagePathToSend</img>")
            }
            prompt.append("<audio>$wavPath</audio>")

            withContext(Dispatchers.Main) {
                chatAdapter.addMessage(ChatMessage("Thinking...", false))
                scrollToBottom()
            }

            nativeChat(prompt.toString())
        }
    }

    fun onChatStreamUpdate(chunk: String) {
        runOnUiThread {
            chatAdapter.updateLastAiMessage(chunk)
            scrollToBottom()
            ttsManager.speak(chunk)
        }
    }

    private fun scrollToBottom() {
        if (chatAdapter.itemCount > 0) {
            binding.chatRecyclerView.smoothScrollToPosition(chatAdapter.itemCount - 1)
        }
    }

    private fun isPermissionGranted(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(this, permission) ==
                PackageManager.PERMISSION_GRANTED
    }

    private fun checkAndRequestStoragePermission(onGranted: () -> Unit) {
        val permission =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
                        Manifest.permission.READ_MEDIA_IMAGES
                else Manifest.permission.READ_EXTERNAL_STORAGE
        if (isPermissionGranted(permission)) onGranted()
        else requestPermissionLauncher.launch(arrayOf(permission))
    }

    private fun checkAndInitModel() {
        if (isModelLoaded || loadingModel) return

        val cacheFile = File(downloadManager.cacheDir)
        if (!cacheFile.exists()) {
            val success = cacheFile.mkdirs()
            Timber.d("Created cache directory: ${cacheFile.absolutePath}, success: $success")
        }

        val info = downloadManager.getDownloadInfo(omniModelId)
        val file = downloadManager.getDownloadedFile(omniModelId)

        if (info.isComplete() && file != null && isValidModelDir(file)) {
            Timber.d("Model is complete and valid at: ${file.absolutePath}")
            // Log file sizes for debugging integrity
            file.listFiles()?.forEach { Timber.d("  - ${it.name}: ${it.length()} bytes") }
            actualModelPath = file.absolutePath
            initModel()
        } else {
            if (file != null && file.exists() && !isValidModelDir(file)) {
                Timber.w(
                        "Model directory exists but is invalid/incomplete, deleting: ${file.absolutePath}"
                )
                DownloadFileUtils.deleteDirectoryRecursively(file)
            }
            loadingModel = true // Mark as loading so we don't call this again
            binding.btnRecordOrSend.isEnabled = false
            binding.inputMessage.hint = "Downloading model..."
            Timber.d("Starting download for $omniModelId")
            downloadManager.startDownload(omniModelId)
        }
    }

    private fun isValidModelDir(dir: File): Boolean {
        if (!dir.exists() || !dir.isDirectory) return false

        // Success marker check for atomicity
        val successMarker = File(dir, ".success")
        if (!successMarker.exists()) {
            Timber.w("Model directory .success marker missing: ${dir.absolutePath}")
            return false
        }

        // Essential files check for Qwen2.5-Omni-3B-MNN
        val essentialFiles =
                arrayOf(
                        "config.json",
                        "llm.mnn",
                        "llm.mnn.weight",
                        "embeddings_bf16.bin",
                        "tokenizer.txt",
                        "audio.mnn",
                        "audio.mnn.weight",
                        "bigvgan.mnn",
                        "bigvgan.mnn.weight",
                        "dit.mnn",
                        "dit.mnn.weight",
                        "predit.mnn",
                        "predit.mnn.weight"
                )

        for (fileName in essentialFiles) {
            val file = File(dir, fileName)
            if (!file.exists() || file.length() == 0L) {
                Timber.w("Missing or empty essential file: $fileName")
                return false
            }
        }
        return true
    }

    private fun initModel() {
        val path = actualModelPath ?: return
        if (isModelLoaded || loadingModel) return
        loadingModel = true
        lifecycleScope.launch(Dispatchers.IO) {
            val modelPath = if (path.endsWith("/")) path else "$path/"
            Timber.d("Initializing model from: $modelPath")
            val success = nativeInit(modelPath)
            withContext(Dispatchers.Main) {
                isModelLoaded = success
                loadingModel = false
                if (success) {
                    binding.btnRecordOrSend.isEnabled = true
                    binding.inputMessage.hint = "Type or hold to talk..."
                } else {
                    Toast.makeText(this@MainActivity, "Failed to load model", Toast.LENGTH_LONG)
                            .show()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::downloadManager.isInitialized) {
            downloadManager.removeListener(this)
        }
        nativeRelease()
    }

    override fun onDownloadStart(modelId: String) {
        Timber.d("Download started: $modelId")
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        val progress = (downloadInfo.progress * 100).toInt()
        Timber.d("onDownloadProgress: $modelId, progress: $progress%")
        runOnUiThread { binding.inputMessage.hint = "Downloading: $progress%" }
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        Timber.d("Download finished at: $path")
        actualModelPath = path
        runOnUiThread {
            binding.inputMessage.hint = "Loading model..."
            initModel()
        }
    }

    override fun onDownloadFailed(modelId: String, e: Exception) {
        Timber.e(e, "Download failed")
        runOnUiThread { Toast.makeText(this, "Model download failed", Toast.LENGTH_SHORT).show() }
    }

    override fun onDownloadPaused(modelId: String) {}
    override fun onDownloadFileRemoved(modelId: String) {}
    override fun onDownloadTotalSize(modelId: String, totalSize: Long) {}
    override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {}

    private fun copyUriToCache(uri: Uri): String? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val file = File(cacheDir, "temp_image_${System.currentTimeMillis()}.jpg")
            val outputStream: OutputStream = FileOutputStream(file)
            inputStream?.copyTo(outputStream)
            inputStream?.close()
            outputStream.close()
            file.absolutePath
        } catch (e: Exception) {
            Timber.e(e, "Failed to copy image")
            null
        }
    }

    private external fun nativeInit(modelDir: String): Boolean
    private external fun nativeChat(prompt: String)
    private external fun nativeRelease()

    companion object {
        init {
            System.loadLibrary("multimodal_audio_jni")
        }
    }
}
