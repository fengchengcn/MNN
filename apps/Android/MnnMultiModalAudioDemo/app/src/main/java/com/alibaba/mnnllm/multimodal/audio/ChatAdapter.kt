package com.alibaba.mnnllm.multimodal.audio

import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

data class ChatMessage(
        val text: String,
        val isUser: Boolean,
        val imagePath: String? = null,
        val isAudio: Boolean = false
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

    fun updateLastAiMessage(newText: String) {
        if (messages.isNotEmpty()) {
            val lastIndex = messages.size - 1
            val lastMsg = messages[lastIndex]
            if (!lastMsg.isUser) {
                messages[lastIndex] = lastMsg.copy(text = lastMsg.text + newText)
                notifyItemChanged(lastIndex)
            } else {
                // Should not happen if logic is correct, but safe fallback
                addMessage(ChatMessage(newText, false))
            }
        }
    }

    fun getLastMessage(): ChatMessage? {
        if (messages.isEmpty()) return null
        return messages.last()
    }

    class UserViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val chatText: TextView = itemView.findViewById(R.id.chat_text)
        private val chatImage: ImageView = itemView.findViewById(R.id.chat_image)

        fun bind(message: ChatMessage) {
            chatText.text = message.text

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
        }
    }

    class AiViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val chatText: TextView = itemView.findViewById(R.id.chat_text)

        fun bind(message: ChatMessage) {
            chatText.text = message.text
        }
    }
}
