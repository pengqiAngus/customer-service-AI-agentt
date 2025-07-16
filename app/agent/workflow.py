import logging
from typing import Dict, List, Any, Optional, Annotated
from dataclasses import dataclass
import json

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from app.config import config
from app.agent.memory import ConversationMemory
from app.agent.rag import KnowledgeBase
from app.agent.text2sql import Text2SQL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """State management for AI Agent"""
    session_id: str
    user_input: str
    conversation_history: List[Dict[str, Any]]
    memory_context: str
    rag_context: str
    sql_result: Optional[Dict[str, Any]]
    response: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_input": self.user_input,
            "conversation_history": self.conversation_history,
            "memory_context": self.memory_context,
            "rag_context": self.rag_context,
            "sql_result": self.sql_result,
            "response": self.response,
            "metadata": self.metadata
        }

class CustomerServiceAgent:
    """AI Customer Service Agent with LangGraph workflow"""
    
    def __init__(self):
        self.memory = ConversationMemory()
        self.knowledge_base = KnowledgeBase()
        self.text2sql = Text2SQL()
        self.llm = None
        self.workflow = None
        
        self._initialize_components()
        self._build_workflow()
    
    def _initialize_components(self):
        """Initialize LLM and other components"""
        try:
            if config.OPENAI_API_KEY:
                self.llm = ChatOpenAI(
                    api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_BASE_URL,
                    model="gpt-3.5-turbo",
                    temperature=config.AGENT_TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                logger.info("LLM initialized for agent workflow")
            else:
                logger.warning("OpenAI API key not configured. Agent will use fallback responses.")
            
            # Initialize knowledge base index
            if self.knowledge_base.vector_store.count() == 0:
                logger.info("Building knowledge base index...")
                self.knowledge_base.build_index()
                
        except Exception as e:
            logger.error(f"Error initializing agent components: {e}")
    
    def _build_workflow(self):
        """Build LangGraph workflow"""
        try:
            # Define workflow graph
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("load_memory", self._load_memory_node)
            workflow.add_node("analyze_intent", self._analyze_intent_node)
            workflow.add_node("retrieve_knowledge", self._retrieve_knowledge_node)
            workflow.add_node("execute_sql", self._execute_sql_node)
            workflow.add_node("generate_response", self._generate_response_node)
            workflow.add_node("save_memory", self._save_memory_node)
            
            # Define edges
            workflow.set_entry_point("load_memory")
            workflow.add_edge("load_memory", "analyze_intent")
            workflow.add_edge("analyze_intent", "retrieve_knowledge")
            
            # Conditional edge for SQL execution
            workflow.add_conditional_edges(
                "retrieve_knowledge",
                self._should_execute_sql,
                {
                    "execute_sql": "execute_sql",
                    "generate_response": "generate_response"
                }
            )
            
            workflow.add_edge("execute_sql", "generate_response")
            workflow.add_edge("generate_response", "save_memory")
            workflow.add_edge("save_memory", END)
            
            # Compile workflow
            self.workflow = workflow.compile()
            logger.info("Agent workflow built successfully")
            
        except Exception as e:
            logger.error(f"Error building workflow: {e}")
            raise
    
    def _load_memory_node(self, state: AgentState) -> AgentState:
        """Load conversation memory"""
        try:
            # Get conversation history
            history = self.memory.get_conversation_history(state.session_id)
            state.conversation_history = history
            
            # Get recent context
            context = self.memory.get_recent_context(state.session_id, context_length=5)
            state.memory_context = context
            
            logger.debug(f"Loaded memory for session {state.session_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            state.memory_context = ""
            return state
    
    def _analyze_intent_node(self, state: AgentState) -> AgentState:
        """Analyze user intent"""
        try:
            user_input = state.user_input.lower()
            metadata = state.metadata
            
            # Simple intent classification
            intent_keywords = {
                "sql_query": ["查询", "统计", "显示", "查看", "多少", "数量", "订单状态", "客户信息"],
                "faq": ["如何", "怎么", "什么", "为什么", "退款", "配送", "质量"],
                "greeting": ["你好", "hello", "hi", "您好"],
                "complaint": ["投诉", "不满", "问题", "错误", "故障"]
            }
            
            detected_intents = []
            for intent, keywords in intent_keywords.items():
                if any(keyword in user_input for keyword in keywords):
                    detected_intents.append(intent)
            
            # Default to FAQ if no specific intent detected
            if not detected_intents:
                detected_intents = ["faq"]
            
            metadata["detected_intents"] = detected_intents
            metadata["needs_sql"] = "sql_query" in detected_intents and config.ENABLE_TEXT2SQL
            
            state.metadata = metadata
            logger.debug(f"Detected intents: {detected_intents}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            state.metadata["detected_intents"] = ["faq"]
            state.metadata["needs_sql"] = False
            return state
    
    def _retrieve_knowledge_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant knowledge from RAG"""
        try:
            # Get relevant context from knowledge base
            context = self.knowledge_base.get_relevant_context(
                state.user_input, 
                max_context_length=1500
            )
            state.rag_context = context
            
            logger.debug(f"Retrieved RAG context: {len(context)} characters")
            return state
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            state.rag_context = ""
            return state
    
    def _should_execute_sql(self, state: AgentState) -> str:
        """Determine if SQL execution is needed"""
        needs_sql = state.metadata.get("needs_sql", False)
        return "execute_sql" if needs_sql else "generate_response"
    
    def _execute_sql_node(self, state: AgentState) -> AgentState:
        """Execute SQL query if needed"""
        try:
            if not config.ENABLE_TEXT2SQL:
                state.sql_result = None
                return state
            
            # Process SQL query
            sql_result = self.text2sql.process_question(state.user_input)
            state.sql_result = sql_result
            
            logger.debug(f"SQL execution result: {sql_result.get('success', False)}")
            return state
            
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            state.sql_result = {
                "success": False,
                "error": f"SQL执行错误: {str(e)}",
                "sql_query": None,
                "data": None
            }
            return state
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate final response"""
        try:
            if self.llm:
                response = self._generate_llm_response(state)
            else:
                response = self._generate_fallback_response(state)
            
            state.response = response
            logger.debug(f"Generated response: {len(response)} characters")
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.response = "抱歉，我现在无法为您提供帮助。请稍后再试。"
            return state
    
    def _generate_llm_response(self, state: AgentState) -> str:
        """Generate response using LLM"""
        try:
            # Build prompt context
            prompt_parts = []
            
            # System message
            system_prompt = """你是一个专业的客户服务AI助手。请根据提供的上下文信息为用户提供准确、友好的回答。

回答要求：
1. 语言友好、专业
2. 回答准确、有帮助
3. 如果有数据查询结果，请清晰地呈现
4. 保持一致的客服语调"""
            
            prompt_parts.append(f"系统提示: {system_prompt}")
            
            # Add conversation context
            if state.memory_context:
                prompt_parts.append(f"对话历史:\n{state.memory_context}")
            
            # Add knowledge base context
            if state.rag_context:
                prompt_parts.append(f"相关知识:\n{state.rag_context}")
            
            # Add SQL results if available
            if state.sql_result and state.sql_result.get("success"):
                data = state.sql_result.get("data", [])
                if data:
                    sql_query = state.sql_result.get("sql_query", "")
                    prompt_parts.append(f"数据库查询结果 (SQL: {sql_query}):")
                    
                    # Format data for presentation
                    if len(data) <= 10:
                        # Show all data if small
                        formatted_data = json.dumps(data, ensure_ascii=False, indent=2)
                    else:
                        # Show summary if large
                        formatted_data = f"查询返回 {len(data)} 条记录，前3条如下：\n"
                        formatted_data += json.dumps(data[:3], ensure_ascii=False, indent=2)
                    
                    prompt_parts.append(formatted_data)
            elif state.sql_result and not state.sql_result.get("success"):
                error = state.sql_result.get("error", "")
                prompt_parts.append(f"数据查询失败: {error}")
            
            # Add user question
            prompt_parts.append(f"用户问题: {state.user_input}")
            prompt_parts.append("请基于以上信息回答用户问题:")
            
            # Create prompt
            prompt = "\n\n".join(prompt_parts)
            
            # Generate response
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(state)
    
    def _generate_fallback_response(self, state: AgentState) -> str:
        """Generate fallback response without LLM"""
        user_input = state.user_input.lower()
        
        # Simple pattern matching responses
        if any(keyword in user_input for keyword in ["你好", "hello", "hi", "您好"]):
            return "您好！我是AI客服助手，很高兴为您服务。请问有什么可以帮助您的吗？"
        
        elif any(keyword in user_input for keyword in ["退款", "退钱"]):
            return "关于退款，您可以提供订单号，我来为您查询和处理。一般退款会在3-5个工作日内到账。"
        
        elif any(keyword in user_input for keyword in ["订单", "查询"]):
            if state.sql_result and state.sql_result.get("success"):
                data = state.sql_result.get("data", [])
                return f"为您查询到 {len(data)} 条相关记录。如需详细信息，请提供具体的订单号。"
            else:
                return "请提供您的订单号，我来为您查询订单状态。"
        
        elif any(keyword in user_input for keyword in ["配送", "物流"]):
            return "我们提供全国配送服务。一般情况下，同城24小时内送达，跨省3-5天送达。您可以提供订单号查询具体配送信息。"
        
        else:
            # Use RAG context if available
            if state.rag_context:
                return f"根据我们的服务信息：{state.rag_context[:200]}..."
            else:
                return "感谢您的咨询。为了更好地帮助您，请提供更具体的问题描述或联系我们的人工客服。"
    
    def _save_memory_node(self, state: AgentState) -> AgentState:
        """Save conversation to memory"""
        try:
            # Save user message
            self.memory.add_message(
                state.session_id, 
                "user", 
                state.user_input,
                metadata={
                    "intents": state.metadata.get("detected_intents", []),
                    "used_sql": bool(state.sql_result)
                }
            )
            
            # Save assistant response
            self.memory.add_message(
                state.session_id,
                "assistant",
                state.response,
                metadata={
                    "rag_used": bool(state.rag_context),
                    "sql_used": bool(state.sql_result),
                    "sql_success": state.sql_result.get("success", False) if state.sql_result else False
                }
            )
            
            logger.debug(f"Saved conversation to memory for session {state.session_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return state
    
    def process_message(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """Process user message through the workflow"""
        try:
            # Create initial state
            initial_state = AgentState(
                session_id=session_id,
                user_input=user_input,
                conversation_history=[],
                memory_context="",
                rag_context="",
                sql_result=None,
                response="",
                metadata={}
            )
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Return response
            return {
                "success": True,
                "response": final_state.response,
                "session_id": session_id,
                "metadata": {
                    "intents": final_state.metadata.get("detected_intents", []),
                    "used_rag": bool(final_state.rag_context),
                    "used_sql": bool(final_state.sql_result),
                    "sql_result": final_state.sql_result
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "success": False,
                "response": "抱歉，处理您的请求时发生了错误。请稍后再试。",
                "session_id": session_id,
                "error": str(e)
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        try:
            return self.memory.get_session_info(session_id)
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {"error": str(e)}
    
    def clear_session(self, session_id: str) -> bool:
        """Clear session memory"""
        try:
            self.memory.clear_conversation(session_id)
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            return {
                "memory_available": self.memory.redis_client is not None,
                "knowledge_base_ready": self.knowledge_base.vector_store.count() > 0,
                "text2sql_enabled": self.text2sql.enabled,
                "llm_available": self.llm is not None,
                "workflow_ready": self.workflow is not None,
                "knowledge_base_stats": self.knowledge_base.get_stats(),
                "text2sql_status": self.text2sql.get_status()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

# Function calling integration
class AgentFunctions:
    """Function calling integration for the agent"""
    
    def __init__(self, agent: CustomerServiceAgent):
        self.agent = agent
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search knowledge base function"""
        try:
            results = self.agent.knowledge_base.search(query, top_k)
            return {
                "success": True,
                "results": results,
                "query": query
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def execute_database_query(self, question: str) -> Dict[str, Any]:
        """Execute database query function"""
        try:
            result = self.agent.text2sql.process_question(question)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary function"""
        try:
            summary = self.agent.memory.get_conversation_summary(session_id)
            return {
                "success": True,
                "summary": summary,
                "session_id": session_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }