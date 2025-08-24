"""
Chat loader and chunking module for ChatWhiz.
Supports multiple chat formats and provides intelligent chunking strategies.
"""

import os
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd


class ChatMessage:
    """Represents a single chat message."""
    
    def __init__(
        self,
        text: str,
        sender: str,
        timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text.strip()
        self.sender = sender
        self.timestamp = timestamp or datetime.now()
        self.message_id = message_id or self._generate_id()
        self.metadata = metadata or {}
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the message."""
        content = f"{self.sender}:{self.text}:{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'text': self.text,
            'sender': self.sender,
            'timestamp': self.timestamp.isoformat(),
            'message_id': self.message_id,
            'metadata': self.metadata
        }





class ChatLoader:
    """Loads and processes chat files from various formats."""

    def __init__(self):
        """Initialize chat loader."""
        pass
    
    def load_whatsapp_export(self, file_path: str) -> List[ChatMessage]:
        """
        Load WhatsApp chat export (txt format).

        Supports various WhatsApp export formats including Unicode characters.
        """
        messages = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Could not decode file {file_path} with any encoding")

        # Clean up Unicode characters that might interfere with parsing
        # Replace narrow no-break space and other Unicode spaces with regular space
        content = re.sub(r'[\u202f\u00a0\u2009\u200a]', ' ', content)

        # Multiple WhatsApp export patterns to handle different formats
        patterns = [
            # Standard format with am/pm
            r'(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2})\s*(am|pm) - ([^:]+): (.+)',
            # Standard format without am/pm (24h)
            r'(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}) - ([^:]+): (.+)',
            # Alternative date format
            r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}) - ([^:]+): (.+)',
            # With seconds
            r'(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}:\d{2}) - ([^:]+): (.+)',
            # Dot separator
            r'(\d{1,2}\.\d{1,2}\.\d{4}), (\d{1,2}:\d{2}) - ([^:]+): (.+)',
            # Brackets format
            r'\[(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}:\d{2})\] ([^:]+): (.+)',
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
            if matches:
                print(f"Using pattern: {pattern}")
                print(f"Found {len(matches)} matches")
                break
        else:
            print("No matching pattern found")
            return messages

        for match in matches:
            groups = match.groups()

            # Handle different group structures based on pattern
            if len(groups) == 5:  # Pattern with am/pm
                date_str, time_str, ampm, sender, text = groups
                time_str = f"{time_str} {ampm}"
                time_format = "%d/%m/%Y %I:%M %p"
            elif len(groups) == 4:  # Standard pattern
                date_str, time_str, sender, text = groups
                # Determine if it's 24h or 12h format
                if ':' in time_str and len(time_str.split(':')) == 3:  # Has seconds
                    time_format = "%d/%m/%Y %H:%M:%S"
                else:
                    time_format = "%d/%m/%Y %H:%M"
            else:
                continue

            # Skip system messages
            if any(keyword in text.lower() for keyword in [
                'created this group', 'left', 'joined', 'changed the group',
                'media omitted', 'deleted this message', 'missed voice call',
                'missed video call', 'security code changed'
            ]):
                continue

            # Parse timestamp
            try:
                if '.' in date_str:  # Dot separator
                    date_str = date_str.replace('.', '/')

                timestamp = datetime.strptime(f"{date_str} {time_str}", time_format)
            except ValueError:
                try:
                    # Try alternative format (MM/DD/YYYY)
                    timestamp = datetime.strptime(f"{date_str} {time_str}", time_format.replace("%d/%m/%Y", "%m/%d/%Y"))
                except ValueError:
                    timestamp = datetime.now()

            # Clean sender name
            sender = sender.strip()
            if sender in ['You', 'you']:
                sender = 'You'

            messages.append(ChatMessage(
                text=text.strip(),
                sender=sender,
                timestamp=timestamp,
                metadata={'source': 'whatsapp', 'file': os.path.basename(file_path)}
            ))

        return messages
    
    def load_json_chat(self, file_path: str) -> List[ChatMessage]:
        """
        Load chat from JSON format.
        
        Expected format: List of objects with 'text', 'sender', 'timestamp' fields
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for item in data:
            timestamp = None
            if 'timestamp' in item:
                try:
                    timestamp = datetime.fromisoformat(item['timestamp'])
                except (ValueError, TypeError):
                    timestamp = datetime.now()
            
            messages.append(ChatMessage(
                text=item.get('text', ''),
                sender=item.get('sender', 'Unknown'),
                timestamp=timestamp,
                message_id=item.get('id'),
                metadata={
                    'source': 'json',
                    'file': os.path.basename(file_path),
                    **item.get('metadata', {})
                }
            ))
        
        return messages
    
    def load_csv_chat(self, file_path: str) -> List[ChatMessage]:
        """
        Load chat from CSV format.
        
        Expected columns: text, sender, timestamp (optional)
        """
        df = pd.read_csv(file_path)
        
        messages = []
        for _, row in df.iterrows():
            timestamp = None
            if 'timestamp' in df.columns and pd.notna(row['timestamp']):
                try:
                    timestamp = pd.to_datetime(row['timestamp']).to_pydatetime()
                except (ValueError, TypeError):
                    timestamp = datetime.now()
            
            messages.append(ChatMessage(
                text=str(row.get('text', '')),
                sender=str(row.get('sender', 'Unknown')),
                timestamp=timestamp,
                metadata={'source': 'csv', 'file': os.path.basename(file_path)}
            ))
        
        return messages
    
    def load_discord_export(self, file_path: str) -> List[ChatMessage]:
        """
        Load Discord chat export (JSON format from DiscordChatExporter).
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for msg in data.get('messages', []):
            timestamp = None
            if 'timestamp' in msg:
                try:
                    timestamp = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    timestamp = datetime.now()
            
            # Combine content and attachments
            text_parts = []
            if msg.get('content'):
                text_parts.append(msg['content'])
            
            for attachment in msg.get('attachments', []):
                if attachment.get('url'):
                    text_parts.append(f"[Attachment: {attachment.get('fileName', 'file')}]")
            
            text = ' '.join(text_parts)
            if not text:
                continue
            
            messages.append(ChatMessage(
                text=text,
                sender=msg.get('author', {}).get('name', 'Unknown'),
                timestamp=timestamp,
                message_id=msg.get('id'),
                metadata={
                    'source': 'discord',
                    'file': os.path.basename(file_path),
                    'channel': data.get('channel', {}).get('name', 'Unknown')
                }
            ))
        
        return messages
    
    def auto_detect_and_load(self, file_path: str) -> List[ChatMessage]:
        """
        Auto-detect file format and load accordingly.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            # Try Discord format first, then generic JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'messages' in data and 'channel' in data:
                    return self.load_discord_export(file_path)
                else:
                    return self.load_json_chat(file_path)
            except Exception as e:
                print(f"Error loading JSON file: {e}")
                return []
        
        elif file_ext == '.csv':
            return self.load_csv_chat(file_path)
        
        elif file_ext == '.txt':
            # Try WhatsApp format
            return self.load_whatsapp_export(file_path)
        
        else:
            print(f"Unsupported file format: {file_ext}")
            return []
    
    def filter_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Filter messages to keep only those suitable for indexing.

        Args:
            messages: List of chat messages

        Returns:
            List of filtered messages
        """
        if not messages:
            return []

        filtered = []
        for message in messages:
            if self._should_index_message(message):
                filtered.append(message)

        return filtered

    def _should_index_message(self, message: ChatMessage) -> bool:
        """Determine if a message should be indexed."""
        text = message.text.strip()
        text_lower = text.lower()

        # Skip very short messages
        if len(text) < 5:
            return False

        # Skip common system/notification messages
        system_patterns = [
            "messages and calls are end-to-end encrypted",
            "this message was deleted",
            "you deleted this message",
            "missed voice call",
            "missed video call",
            "image omitted",
            "video omitted",
            "audio omitted",
            "document omitted",
            "sticker omitted",
            "gif omitted",
            "location omitted",
            "contact card omitted",
            "poll omitted",
            "waiting for this message",
            "this message was deleted",
            "media omitted"
        ]

        for pattern in system_patterns:
            if pattern in text_lower:
                return False

        # Skip messages that look like AI artifacts or generated content
        ai_patterns = [
            "```",  # Code blocks
            "here's a refined blueprint",
            "here's a",
            "blueprint for",
            "semantic search project",
            "instructor-large",
            "embedding model",
            "# ",  # Markdown headers
            "## ",
            "### "
        ]

        for pattern in ai_patterns:
            if pattern in text_lower:
                return False

        # Skip very repetitive content (less than 3 unique words)
        words = text_lower.split()
        if len(words) > 0 and len(set(words)) < min(3, len(words)):
            return False

        # Skip messages that are just punctuation or symbols
        if all(not c.isalnum() for c in text):
            return False

        # Skip messages that are just numbers or single characters repeated
        if len(text.strip()) <= 2:
            return False

        return True

    def deduplicate_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Remove duplicate messages based on content similarity."""
        seen_texts = set()
        unique_messages = []

        for message in messages:
            # Use a simplified version of the text for deduplication
            simplified_text = re.sub(r'\s+', ' ', message.text.lower().strip())

            if simplified_text not in seen_texts:
                seen_texts.add(simplified_text)
                unique_messages.append(message)

        print(f"Deduplicated {len(messages)} messages to {len(unique_messages)} unique messages")
        return unique_messages
