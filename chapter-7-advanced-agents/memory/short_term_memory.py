"""
Short-term memory for conversation context.
"""

from typing import List, Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ConversationBufferMemory:
    """Simple buffer memory that stores recent conversation history"""
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize conversation buffer
        
        Args:
            max_messages: Maximum number of messages to keep
        """
        self.max_messages = max_messages
        self.messages: deque = deque(maxlen=max_messages)
    
    def add_message(self, role: str, content: str):
        """Add message to buffer"""
        self.messages.append({
            "role": role,
            "content": content
        })
    
    def add_user_message(self, content: str):
        """Add user message"""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str):
        """Add assistant message"""
        self.add_message("assistant", content)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in buffer"""
        return list(self.messages)
    
    def get_context_string(self) -> str:
        """Get messages formatted as string"""
        context = []
        for msg in self.messages:
            context.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(context)
    
    def clear(self):
        """Clear all messages"""
        self.messages.clear()
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """Get last N messages"""
        return list(self.messages)[-n:]


class SlidingWindowMemory(ConversationBufferMemory):
    """Memory with sliding window that keeps recent context"""
    
    def __init__(self, max_tokens: int = 2000, tokens_per_message: int = 100):
        """
        Initialize sliding window memory
        
        Args:
            max_tokens: Maximum tokens to keep in context
            tokens_per_message: Estimated tokens per message
        """
        max_messages = max_tokens // tokens_per_message
        super().__init__(max_messages=max_messages)
        self.max_tokens = max_tokens
        self.tokens_per_message = tokens_per_message
    
    def add_message(self, role: str, content: str):
        """Add message and manage token count"""
        # Estimate tokens in this message
        estimated_tokens = len(content.split()) * 1.3  # rough estimate
        
        # Remove old messages if needed to stay under limit
        total_tokens = sum(len(m['content'].split()) * 1.3 for m in self.messages)
        
        while total_tokens + estimated_tokens > self.max_tokens and self.messages:
            removed = self.messages.popleft()
            total_tokens -= len(removed['content'].split()) * 1.3
        
        # Add new message
        super().add_message(role, content)


class ConversationSummaryMemory:
    """Memory that summarizes old conversations to save context"""
    
    def __init__(self, llm_client, max_messages: int = 10):
        """
        Initialize summary memory
        
        Args:
            llm_client: LLM client for generating summaries
            max_messages: Messages to keep before summarizing
        """
        self.llm = llm_client
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []
        self.summary: Optional[str] = None
    
    def add_message(self, role: str, content: str):
        """Add message and summarize if needed"""
        self.messages.append({"role": role, "content": content})
        
        if len(self.messages) >= self.max_messages:
            self._summarize_and_compact()
    
    def _summarize_and_compact(self):
        """Summarize old messages and keep recent ones"""
        # Keep last 3 messages, summarize the rest
        messages_to_summarize = self.messages[:-3]
        
        if not messages_to_summarize:
            return
        
        # Create conversation text
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages_to_summarize
        ])
        
        # Generate summary
        prompt = f"""Summarize this conversation, preserving key information:

{conversation_text}

Provide a concise summary that captures the main points and context."""
        
        try:
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            new_summary = response.content[0].text
            
            # Append to existing summary
            if self.summary:
                self.summary = f"{self.summary}\n\n{new_summary}"
            else:
                self.summary = new_summary
            
            # Keep only recent messages
            self.messages = self.messages[-3:]
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages with summary as context"""
        if self.summary:
            return [
                {"role": "system", "content": f"Previous conversation summary:\n{self.summary}"}
            ] + self.messages
        return self.messages
    
    def get_context_string(self) -> str:
        """Get formatted context with summary"""
        context = []
        
        if self.summary:
            context.append(f"Summary of previous conversation:\n{self.summary}\n")
        
        for msg in self.messages:
            context.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context)
    
    def clear(self):
        """Clear memory and summary"""
        self.messages.clear()
        self.summary = None