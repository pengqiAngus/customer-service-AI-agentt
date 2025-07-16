import redis
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationMemory:
    """使用 Redis 管理对话记忆"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self.redis_client = None
            self._memory_store = {}  # Fallback in-memory store
    
    def _get_conversation_key(self, session_id: str) -> str:
        """生成 Redis 对话键名"""
        return f"conversation:{session_id}"
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息到对话历史"""
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            if self.redis_client:
                # Redis storage
                key = self._get_conversation_key(session_id)
                self.redis_client.lpush(key, json.dumps(message))
                
                # Keep only recent messages
                self.redis_client.ltrim(key, 0, config.MAX_CONVERSATION_HISTORY - 1)
                
                # Set expiration
                self.redis_client.expire(key, config.MEMORY_TTL)
            else:
                # In-memory fallback
                if session_id not in self._memory_store:
                    self._memory_store[session_id] = []
                
                self._memory_store[session_id].insert(0, message)
                
                # Keep only recent messages
                if len(self._memory_store[session_id]) > config.MAX_CONVERSATION_HISTORY:
                    self._memory_store[session_id] = self._memory_store[session_id][:config.MAX_CONVERSATION_HISTORY]
            
            logger.debug(f"Added message to session {session_id}: {role}")
            
        except Exception as e:
            logger.error(f"Error adding message to memory: {e}")
    
    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取会话的对话历史"""
        try:
            if limit is None:
                limit = config.MAX_CONVERSATION_HISTORY
            
            if self.redis_client:
                # Redis storage
                key = self._get_conversation_key(session_id)
                messages = self.redis_client.lrange(key, 0, limit - 1)
                history = [json.loads(msg) for msg in messages]
                # Reverse to get chronological order
                return list(reversed(history))
            else:
                # In-memory fallback
                if session_id in self._memory_store:
                    history = self._memory_store[session_id][:limit]
                    return list(reversed(history))
                return []
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def get_recent_context(self, session_id: str, context_length: int = 5) -> str:
        """获取最近的对话上下文（格式化字符串）"""
        try:
            history = self.get_conversation_history(session_id, limit=context_length)
            
            if not history:
                return ""
            
            context_parts = []
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"用户: {content}")
                elif role == "assistant":
                    context_parts.append(f"客服: {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting recent context: {e}")
            return ""
    
    def clear_conversation(self, session_id: str):
        """清除某会话的对话历史"""
        try:
            if self.redis_client:
                key = self._get_conversation_key(session_id)
                self.redis_client.delete(key)
            else:
                if session_id in self._memory_store:
                    del self._memory_store[session_id]
            
            logger.info(f"Cleared conversation for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息和统计"""
        try:
            history = self.get_conversation_history(session_id)
            
            if not history:
                return {
                    "session_id": session_id,
                    "message_count": 0,
                    "last_activity": None,
                    "session_active": False
                }
            
            user_messages = len([msg for msg in history if msg.get("role") == "user"])
            assistant_messages = len([msg for msg in history if msg.get("role") == "assistant"])
            last_message = history[-1] if history else None
            
            return {
                "session_id": session_id,
                "message_count": len(history),
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "last_activity": last_message.get("timestamp") if last_message else None,
                "session_active": True,
                "last_message_role": last_message.get("role") if last_message else None
            }
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {
                "session_id": session_id,
                "message_count": 0,
                "last_activity": None,
                "session_active": False,
                "error": str(e)
            }
    
    def search_conversations(self, session_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索包含关键词的消息"""
        try:
            history = self.get_conversation_history(session_id)
            
            matching_messages = []
            for msg in history:
                content = msg.get("content", "").lower()
                if query.lower() in content:
                    matching_messages.append(msg)
                    if len(matching_messages) >= limit:
                        break
            
            return matching_messages
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """生成会话摘要"""
        try:
            history = self.get_conversation_history(session_id)
            
            if not history:
                return {"summary": "No conversation history", "topics": []}
            
            # Extract topics and keywords (simple implementation)
            all_content = " ".join([msg.get("content", "") for msg in history])
            
            # Common customer service topics
            topics = []
            topic_keywords = {
                "退款": ["退款", "退钱", "返钱"],
                "订单": ["订单", "下单", "购买"],
                "配送": ["配送", "快递", "物流", "送货"],
                "产品": ["产品", "商品", "质量"],
                "客服": ["客服", "服务", "帮助"]
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in all_content for keyword in keywords):
                    topics.append(topic)
            
            return {
                "summary": f"会话包含 {len(history)} 条消息，主要讨论了客户服务相关问题",
                "topics": topics,
                "message_count": len(history),
                "duration": "活跃会话"
            }
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return {"summary": "无法生成摘要", "topics": [], "error": str(e)}