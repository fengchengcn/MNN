package com.alibaba.mnnllm.multimodal.audio

import android.graphics.BitmapFactory
import android.media.MediaPlayer
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

data class ChatMessage(
        val text: String,
        val isUser: Boolean,
        val imagePath: String? = null,
        val isAudio: Boolean = false,
        val audioPath: String? = null
)

class ChatAdapter(private val messages: MutableList<ChatMessage> = mutableListOf()) :
        RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    companion object {
        private const val TYPE_USER = 1
        private const val TYPE_AI = 2
    }

    override fun getItemViewType(position: Int): Int {
        return if (messages[position].isUser) TYPE_USER else TYPE_AI
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val layoutInflater = LayoutInflater.from(parent.context)
        return if (viewType == TYPE_USER) {
            val view = layoutInflater.inflate(R.layout.item_chat_user, parent, false)
            UserViewHolder(view)
        } else {
            val view = layoutInflater.inflate(R.layout.item_chat_ai, parent, false)
            AiViewHolder(view)
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]
        if (holder is UserViewHolder) {
            holder.bind(message)
        } else if (holder is AiViewHolder) {
            holder.bind(message)
        }
    }

    override fun getItemCount(): Int = messages.size

    fun addMessage(message: ChatMessage) {
        messages.add(message)
        notifyItemInserted(messages.size - 1)
    }

    fun clear() {
        val size = messages.size
        messages.clear()
        notifyItemRangeRemoved(0, size)
    }

    fun updateLastAiMessage(newText: String) {
        if (messages.isNotEmpty()) {
            val lastIndex = messages.size - 1
            val lastMsg = messages[lastIndex]
            if (!lastMsg.isUser) {
                messages[lastIndex] = lastMsg.copy(text = lastMsg.text + newText)
                notifyItemChanged(lastIndex)
            } else {
                addMessage(ChatMessage(newText, false))
            }
        }
    }

    fun replaceLastMessage(newText: String, isUser: Boolean) {
        if (messages.isNotEmpty()) {
            val lastIndex = messages.size - 1
            messages[lastIndex] = ChatMessage(newText, isUser)
            notifyItemChanged(lastIndex)
        } else {
            addMessage(ChatMessage(newText, isUser))
        }
    }

    fun getLastMessage(): ChatMessage? {
        if (messages.isEmpty()) return null
        return messages.last()
    }

    class UserViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val chatText: TextView = itemView.findViewById(R.id.chat_text)
        private val chatImage: ImageView = itemView.findViewById(R.id.chat_image)
        private val audioLayout: LinearLayout = itemView.findViewById(R.id.audio_layout)
        private val btnPlayAudio: ImageView = itemView.findViewById(R.id.btn_play_audio)

        companion object {
            private var currentMediaPlayer: MediaPlayer? = null
            private var currentPlayingButton: ImageView? = null
            private var currentPlayingPath: String? = null
        }

        fun bind(message: ChatMessage) {
            chatText.text = if (message.isAudio) "" else message.text
            chatText.visibility =
                    if (message.isAudio && message.text == "[Audio Message]") View.GONE
                    else View.VISIBLE

            if (message.imagePath != null) {
                chatImage.visibility = View.VISIBLE
                try {
                    val bitmap = BitmapFactory.decodeFile(message.imagePath)
                    chatImage.setImageBitmap(bitmap)
                } catch (e: Exception) {
                    chatImage.visibility = View.GONE
                }
            } else {
                chatImage.visibility = View.GONE
            }

            if (message.isAudio && message.audioPath != null) {
                audioLayout.visibility = View.VISIBLE
                btnPlayAudio.setOnClickListener { playAudio(message.audioPath) }

                if (currentPlayingPath == message.audioPath && currentMediaPlayer?.isPlaying == true
                ) {
                    btnPlayAudio.setImageResource(android.R.drawable.ic_media_pause)
                    currentPlayingButton = btnPlayAudio
                } else {
                    btnPlayAudio.setImageResource(android.R.drawable.ic_media_play)
                }
            } else {
                audioLayout.visibility = View.GONE
            }
        }

        private fun playAudio(path: String) {
            try {
                if (currentPlayingPath == path && currentMediaPlayer?.isPlaying == true) {
                    currentMediaPlayer?.stop()
                    btnPlayAudio.setImageResource(android.R.drawable.ic_media_play)
                    currentPlayingPath = null
                    currentPlayingButton = null
                    return
                }

                currentMediaPlayer?.release()
                currentPlayingButton?.setImageResource(android.R.drawable.ic_media_play)

                currentMediaPlayer =
                        MediaPlayer().apply {
                            setDataSource(path)
                            prepare()
                            start()
                        }
                currentPlayingPath = path
                currentPlayingButton = btnPlayAudio
                btnPlayAudio.setImageResource(android.R.drawable.ic_media_pause)

                currentMediaPlayer?.setOnCompletionListener {
                    btnPlayAudio.setImageResource(android.R.drawable.ic_media_play)
                    if (currentPlayingPath == path) {
                        currentPlayingPath = null
                        currentPlayingButton = null
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    class AiViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val chatText: TextView = itemView.findViewById(R.id.chat_text)

        fun bind(message: ChatMessage) {
            chatText.text = message.text
        }
    }
}
