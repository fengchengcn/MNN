package com.alibaba.mnnllm.multimodal.audio

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mnnllm.multimodal.audio.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Example of a call to a native method
        binding.sampleText.text = stringFromJNI()
    }

    /**
     * A native method that is implemented by the 'mnn_multimodal_audio' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'mnn_multimodal_audio' library on application startup.
        init {
            System.loadLibrary("mnn_multimodal_audio")
        }
    }
}
