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
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.widget.addTextChangedListener
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
    private var isVoiceMode = false
    private var isFirstChunk = true

    // Model ID for download manager
    private val omniModelId = "ModelScope/MNN/Qwen2.5-Omni-3B-MNN"

    // Persistent storage path
    private val PERSISTENT_CACHE_DIR =
            Environment.getExternalStorageDirectory().absolutePath + "/MnnModels"

    private val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
                    permissions ->
                val recordGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
                if (permissions.containsKey(Manifest.permission.RECORD_AUDIO) && !recordGranted) {
                    Toast.makeText(this, "Need Audio Permission", Toast.LENGTH_SHORT).show()
                }

                val imagePermission =
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
                                Manifest.permission.READ_MEDIA_IMAGES
                        else Manifest.permission.READ_EXTERNAL_STORAGE

                if (permissions[imagePermission] == true) {
                    openGallery()
                }
            }

    private val pickMedia =
            registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
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
        audioHandler.onAmplitudeUpdate = { amplitude ->
            runOnUiThread { binding.waveformView.addAmplitude(amplitude) }
        }
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
            checkAndRequestStoragePermission { openGallery() }
        }

        binding.btnClosePreview.setOnClickListener { clearImagePreview() }

        binding.inputMessage.addTextChangedListener {
            val hasText = it?.toString()?.trim()?.isNotEmpty() == true
            if (!isVoiceMode) {
                binding.btnSend.visibility =
                        if (hasText || currentImagePath != null) View.VISIBLE else View.GONE
            }
        }

        binding.btnToggleVoice.setOnClickListener {
            isVoiceMode = !isVoiceMode
            updateUIForMode()
        }

        binding.btnClear.setOnClickListener { clearChat() }

        binding.btnSend.setOnClickListener {
            val text = binding.inputMessage.text.toString().trim()
            if (text.isNotEmpty() || currentImagePath != null) {
                handleSendText(text)
            }
        }

        binding.btnHoldToTalk.setOnTouchListener { v, event ->
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
                    v.isPressed = true
                    v.performHapticFeedback(android.view.HapticFeedbackConstants.LONG_PRESS)
                    ttsManager.stop()
                    startLevelRecording()
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    v.isPressed = false
                    stopLevelRecording()
                    true
                }
                else -> false
            }
        }
    }

    private fun updateUIForMode() {
        if (isVoiceMode) {
            binding.btnToggleVoice.setImageResource(
                    android.R.drawable.ic_menu_edit
            ) // Keyboard icon
            binding.inputMessage.visibility = View.GONE
            binding.btnSend.visibility = View.GONE
            binding.btnHoldToTalk.visibility = View.VISIBLE
        } else {
            binding.btnToggleVoice.setImageResource(R.drawable.ic_mic)
            binding.inputMessage.visibility = View.VISIBLE
            binding.btnHoldToTalk.visibility = View.GONE
            val hasText = binding.inputMessage.text.toString().trim().isNotEmpty()
            binding.btnSend.visibility = if (hasText) View.VISIBLE else View.GONE
        }
    }

    private fun startLevelRecording() {
        audioHandler.startRecording()
        showRecordingIndicator(true)
    }

    private fun stopLevelRecording() {
        showRecordingIndicator(false)
        val wavPath = audioHandler.stopRecording()
        if (wavPath != null) {
            processAudio(wavPath)
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
        if (!isVoiceMode) {
            binding.btnSend.visibility = View.VISIBLE
        }
    }

    private fun clearImagePreview() {
        currentImagePath = null
        binding.previewCard.visibility = View.GONE
        binding.imgPreviewMini.setImageDrawable(null)
    }

    private fun showRecordingIndicator(show: Boolean) {
        binding.recordingIndicator.visibility = if (show) View.VISIBLE else View.GONE
        if (!show) {
            binding.waveformView.clear()
        }
    }

    private fun handleSendText(text: String) {
        if (!isModelLoaded) {
            Toast.makeText(this, "Model is loading...", Toast.LENGTH_SHORT).show()
            return
        }

        val message = ChatMessage(text, true, currentImagePath)
        chatAdapter.addMessage(message)
        scrollToBottom()

        val imagePathToSend = currentImagePath
        clearImagePreview()
        binding.inputMessage.text.clear()

        isFirstChunk = true
        processText(text, imagePathToSend)
    }

    private fun clearChat() {
        chatAdapter.clear()
        conversationHistory.clear()
        nativeReset()
        clearImagePreview()
        binding.inputMessage.text.clear()
        Toast.makeText(this, "Chat cleared", Toast.LENGTH_SHORT).show()
    }

    private fun processText(text: String, imagePath: String?) {
        lifecycleScope.launch(Dispatchers.IO) {
            val prompt = StringBuilder()
            if (imagePath != null) {
                prompt.append("<img>$imagePath</img>")
            }
            if (text.isNotEmpty()) {
                prompt.append(text)
            } else if (imagePath != null) {
                prompt.append("Describe this image.")
            }

            val finalPrompt = prompt.toString()
            Timber.i("Processing Text Prompt: $finalPrompt")

            withContext(Dispatchers.Main) {
                chatAdapter.addMessage(ChatMessage("Thinking...", false))
                scrollToBottom()
            }

            startChat(finalPrompt)
        }
    }

    private fun processAudio(wavPath: String) {
        if (!isModelLoaded) {
            Toast.makeText(this, "Model is loading...", Toast.LENGTH_SHORT).show()
            return
        }

        val imagePathToSend = currentImagePath
        chatAdapter.addMessage(ChatMessage("[Audio Message]", true, imagePathToSend, true, wavPath))
        scrollToBottom()
        clearImagePreview()

        lifecycleScope.launch(Dispatchers.IO) {
            val prompt = StringBuilder()
            if (imagePathToSend != null) {
                prompt.append("<img>$imagePathToSend</img>")
            }
            prompt.append("<audio>$wavPath</audio>")
            prompt.append("请把这段录音转写成文字。")

            val finalPrompt = prompt.toString()
            Timber.i("Processing Audio Prompt: $finalPrompt")

            isFirstChunk = true
            withContext(Dispatchers.Main) {
                chatAdapter.addMessage(ChatMessage("Thinking...", false))
                scrollToBottom()
            }

            startChat(finalPrompt)
        }
    }

    private val conversationHistory = mutableListOf<Pair<String, String>>()
    private val MAX_HISTORY_TURNS = 5 // Keep last 5 turns (10 messages)

    private fun addMessageToHistory(role: String, content: String) {
        conversationHistory.add(role to content)
        while (conversationHistory.size > MAX_HISTORY_TURNS * 2) {
            conversationHistory.removeAt(0)
        }
    }

    private fun startChat(prompt: String) {
        addMessageToHistory("user", prompt)

        // Pass only the current turn (User + Prompt) to nativeChat.
        // The LLM maintains its own KV cache / history state internally.
        // Attempting to re-send file paths from history causes "File not found" errors and
        // re-processing overhead.
        val currentTurn = arrayOf("user", prompt)
        nativeChat(currentTurn)
    }

    fun onChatStreamUpdate(chunk: String) {
        runOnUiThread {
            if (isFirstChunk) {
                chatAdapter.replaceLastMessage(chunk, false)
                isFirstChunk = false
            } else {
                chatAdapter.updateLastAiMessage(chunk)
            }
            scrollToBottom()
            ttsManager.speak(chunk)
        }
    }

    fun onChatFinished(fullResponse: String) {
        runOnUiThread {
            Timber.i("Chat finished. Full response: $fullResponse")
            addMessageToHistory("assistant", fullResponse)
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
        // For modern Android (11+), the system photo picker doesn't need permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            onGranted()
            return
        }

        val permission = Manifest.permission.READ_EXTERNAL_STORAGE
        if (isPermissionGranted(permission)) {
            onGranted()
        } else {
            requestPermissionLauncher.launch(arrayOf(permission))
        }
    }

    private fun openGallery() {
        pickMedia.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly))
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
                    binding.inputMessage.hint = "Type in ..."
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
    private external fun nativeChat(history: Array<String>)
    private external fun nativeReset()
    private external fun nativeRelease()

    companion object {
        init {
            System.loadLibrary("multimodal_audio_jni")
        }
    }
}
