"""
Data API module for Supabase integration.
This file contains database operations for the LAISA chatbot.
"""

import os
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from loguru import logger


class DataAPI:
    """Handles database operations for conversation storage and retrieval."""
    
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.client: Optional[Client] = None
        
        if self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
        else:
            logger.warning("Supabase credentials not found in environment variables")
    
    def is_connected(self) -> bool:
        """Check if database connection is available."""
        return self.client is not None
    
    def insert_conversation(self, conversation_data: List[Dict[str, Any]], metadata: str = "conversation") -> bool:
        """
        Insert conversation data into the database.
        
        Args:
            conversation_data: List of conversation messages
            metadata: Metadata tag for the conversation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Database not connected")
            return False
        
        try:
            json_value = {
                'meta': metadata, 
                'conversation': conversation_data
            }
            
            data = self.client.table('data_meta').insert(json_value).execute()
            logger.info("Conversation saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert conversation: {e}")
            return False
    
    def fetch_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch conversations from the database.
        
        Args:
            limit: Maximum number of conversations to fetch
            
        Returns:
            List of conversation records
        """
        if not self.is_connected():
            logger.error("Database not connected")
            return []
        
        try:
            response = self.client.table("data_meta").select("*").limit(limit).execute()
            logger.info(f"Fetched {len(response.data)} conversations")
            return response.data
            
        except Exception as e:
            logger.error(f"Failed to fetch conversations: {e}")
            return []


# Global instance for backward compatibility
_data_api = DataAPI()

def get_data_api() -> DataAPI:
    """Get the global DataAPI instance."""
    return _data_api
