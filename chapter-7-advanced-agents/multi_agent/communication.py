"""
Agent-to-agent communication protocols.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    QUERY = "query"


@dataclass
class Message:
    """Message between agents"""
    sender: str
    recipient: str
    message_type: MessageType
    content: Any
    message_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    in_reply_to: Optional[str] = None


class MessageBus:
    """Central message bus for agent communication"""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.subscribers: Dict[str, List[str]] = {}  # topic -> list of agent names
        self.agent_mailboxes: Dict[str, List[Message]] = {}
    
    def send_message(self, message: Message):
        """
        Send message through the bus
        
        Args:
            message: Message to send
        """
        self.messages.append(message)
        
        # Deliver to recipient's mailbox
        if message.recipient not in self.agent_mailboxes:
            self.agent_mailboxes[message.recipient] = []
        
        self.agent_mailboxes[message.recipient].append(message)
        
        logger.info(f"Message sent: {message.sender} -> {message.recipient} ({message.message_type.value})")
    
    def broadcast_message(self, sender: str, content: Any, topic: Optional[str] = None):
        """
        Broadcast message to all subscribers
        
        Args:
            sender: Sender agent name
            content: Message content
            topic: Optional topic filter
        """
        recipients = self.subscribers.get(topic, []) if topic else list(self.agent_mailboxes.keys())
        
        for recipient in recipients:
            if recipient != sender:  # Don't send to self
                message = Message(
                    sender=sender,
                    recipient=recipient,
                    message_type=MessageType.BROADCAST,
                    content=content,
                    metadata={'topic': topic} if topic else {}
                )
                self.send_message(message)
    
    def subscribe(self, agent_name: str, topic: str):
        """
        Subscribe agent to topic
        
        Args:
            agent_name: Agent name
            topic: Topic to subscribe to
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        
        if agent_name not in self.subscribers[topic]:
            self.subscribers[topic].append(agent_name)
            logger.info(f"Agent {agent_name} subscribed to topic: {topic}")
    
    def unsubscribe(self, agent_name: str, topic: str):
        """Unsubscribe agent from topic"""
        if topic in self.subscribers and agent_name in self.subscribers[topic]:
            self.subscribers[topic].remove(agent_name)
            logger.info(f"Agent {agent_name} unsubscribed from topic: {topic}")
    
    def get_messages(self, agent_name: str, message_type: Optional[MessageType] = None) -> List[Message]:
        """
        Get messages for agent
        
        Args:
            agent_name: Agent name
            message_type: Optional filter by message type
            
        Returns:
            List of messages
        """
        if agent_name not in self.agent_mailboxes:
            return []
        
        messages = self.agent_mailboxes[agent_name]
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        return messages
    
    def clear_mailbox(self, agent_name: str):
        """Clear agent's mailbox"""
        if agent_name in self.agent_mailboxes:
            self.agent_mailboxes[agent_name] = []
    
    def get_conversation(self, agent1: str, agent2: str) -> List[Message]:
        """Get conversation between two agents"""
        conversation = []
        
        for message in self.messages:
            if (message.sender == agent1 and message.recipient == agent2) or \
               (message.sender == agent2 and message.recipient == agent1):
                conversation.append(message)
        
        return sorted(conversation, key=lambda m: m.timestamp)


class CommunicationProtocol:
    """Protocol for structured agent communication"""
    
    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
    
    def request_information(self, requester: str, provider: str, query: str) -> str:
        """
        Request information from another agent
        
        Args:
            requester: Requesting agent name
            provider: Agent to ask
            query: Information query
            
        Returns:
            Message ID for tracking
        """
        message = Message(
            sender=requester,
            recipient=provider,
            message_type=MessageType.QUERY,
            content={'query': query}
        )
        
        self.bus.send_message(message)
        return message.message_id
    
    def respond_to_request(self, responder: str, original_message_id: str, response_content: Any):
        """
        Respond to a request
        
        Args:
            responder: Responding agent name
            original_message_id: ID of original request message
            response_content: Response content
        """
        # Find original message
        original = None
        for msg in self.bus.messages:
            if msg.message_id == original_message_id:
                original = msg
                break
        
        if not original:
            logger.warning(f"Original message {original_message_id} not found")
            return
        
        response = Message(
            sender=responder,
            recipient=original.sender,
            message_type=MessageType.RESPONSE,
            content=response_content,
            in_reply_to=original_message_id
        )
        
        self.bus.send_message(response)
    
    def notify_agents(self, sender: str, recipients: List[str], notification: str):
        """
        Send notification to multiple agents
        
        Args:
            sender: Sender agent name
            recipients: List of recipient agent names
            notification: Notification message
        """
        for recipient in recipients:
            message = Message(
                sender=sender,
                recipient=recipient,
                message_type=MessageType.NOTIFICATION,
                content={'notification': notification}
            )
            self.bus.send_message(message)