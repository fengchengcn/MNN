package com.alibaba.mnnllm.multimodal.audio

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class WaveformView
@JvmOverloads
constructor(context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0) :
        View(context, attrs, defStyleAttr) {

    private val paint =
            Paint().apply {
                color = Color.WHITE
                style = Paint.Style.FILL
                isAntiAlias = true
            }

    private val amplitudes = mutableListOf<Float>()
    private val maxBars = 40
    private val barGap = 4f
    private var barWidth = 8f

    fun addAmplitude(amplitude: Int) {
        // Normalize amplitude (assuming max is around 32767 for 16-bit PCM)
        val normalized = (amplitude.toFloat() / 32767f).coerceIn(0.1f, 1.0f)
        amplitudes.add(normalized)
        if (amplitudes.size > maxBars) {
            amplitudes.removeAt(0)
        }
        invalidate()
    }

    fun clear() {
        amplitudes.clear()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (amplitudes.isEmpty()) return

        val w = width.toFloat()
        val h = height.toFloat()
        val centerY = h / 2f

        barWidth = (w - (maxBars - 1) * barGap) / maxBars

        for (i in amplitudes.indices) {
            val amp = amplitudes[i]
            val barHeight = amp * h * 0.8f
            val left = i * (barWidth + barGap)
            val top = centerY - barHeight / 2f
            val right = left + barWidth
            val bottom = centerY + barHeight / 2f

            canvas.drawRoundRect(left, top, right, bottom, barWidth / 2f, barWidth / 2f, paint)
        }
    }
}
