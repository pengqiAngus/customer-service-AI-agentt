from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, Namespace
import uuid
import logging
from typing import Dict, Any

from app.config import config
from app.agent.workflow import CustomerServiceAgent, AgentFunctions
from app.models.trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-customer-service-secret-key'

# Initialize Flask-RESTX with Swagger UI
api = Api(
    app,
    version='1.0',
    title='AI客户服务Agent API',
    description='基于LangChain和LangGraph的AI客户服务Agent系统',
    doc='/docs/',
    prefix='/api/v1'
)

# Initialize agent
try:
    agent = CustomerServiceAgent()
    agent_functions = AgentFunctions(agent)
    logger.info("AI Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI Agent: {e}")
    agent = None
    agent_functions = None

# Define namespaces
chat_ns = Namespace('chat', description='聊天对话相关接口')
training_ns = Namespace('training', description='模型训练相关接口')
knowledge_ns = Namespace('knowledge', description='知识库相关接口')
sql_ns = Namespace('sql', description='Text2SQL相关接口')
system_ns = Namespace('system', description='系统状态相关接口')

api.add_namespace(chat_ns)
api.add_namespace(training_ns)
api.add_namespace(knowledge_ns)
api.add_namespace(sql_ns)
api.add_namespace(system_ns)

# Request/Response models
chat_request = api.model('ChatRequest', {
    'message': fields.String(required=True, description='用户消息'),
    'session_id': fields.String(required=False, description='会话ID，不提供则自动生成')
})

chat_response = api.model('ChatResponse', {
    'success': fields.Boolean(description='请求是否成功'),
    'response': fields.String(description='AI回复'),
    'session_id': fields.String(description='会话ID'),
    'metadata': fields.Raw(description='元数据信息'),
    'error': fields.String(description='错误信息')
})

training_request = api.model('TrainingRequest', {
    'rebuild_model': fields.Boolean(default=False, description='是否重新训练模型')
})

knowledge_request = api.model('KnowledgeRequest', {
    'query': fields.String(required=True, description='搜索查询'),
    'top_k': fields.Integer(default=5, description='返回结果数量')
})

sql_request = api.model('SQLRequest', {
    'question': fields.String(required=True, description='自然语言问题')
})

# Chat endpoints
@chat_ns.route('/message')
class ChatMessage(Resource):
    @chat_ns.expect(chat_request)
    @chat_ns.marshal_with(chat_response)
    @chat_ns.doc('send_message', description='发送消息给AI助手')
    def post(self):
        """发送消息给AI客服助手"""
        try:
            if not agent:
                return {
                    'success': False,
                    'response': 'AI助手暂时不可用',
                    'error': 'Agent not initialized'
                }, 500
            
            data = request.get_json()
            if not data or 'message' not in data:
                return {
                    'success': False,
                    'response': '消息内容不能为空',
                    'error': 'Message is required'
                }, 400
            
            message = data['message']
            session_id = data.get('session_id') or str(uuid.uuid4())
            
            # Process message
            result = agent.process_message(session_id, message)
            
            return result, 200 if result['success'] else 500
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                'success': False,
                'response': '处理消息时发生错误',
                'error': str(e)
            }, 500

@chat_ns.route('/session/<string:session_id>')
class ChatSession(Resource):
    @chat_ns.doc('get_session_info', description='获取会话信息')
    def get(self, session_id):
        """获取会话信息"""
        try:
            if not agent:
                return {'error': 'Agent not initialized'}, 500
            
            session_info = agent.get_session_info(session_id)
            return session_info, 200
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {'error': str(e)}, 500
    
    @chat_ns.doc('clear_session', description='清除会话记录')
    def delete(self, session_id):
        """清除会话记录"""
        try:
            if not agent:
                return {'error': 'Agent not initialized'}, 500
            
            success = agent.clear_session(session_id)
            return {
                'success': success,
                'message': '会话已清除' if success else '清除会话失败'
            }, 200 if success else 500
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return {'error': str(e)}, 500

# Training endpoints
@training_ns.route('/start')
class TrainingStart(Resource):
    @training_ns.expect(training_request)
    @training_ns.doc('start_training', description='开始模型训练')
    def post(self):
        """开始模型训练"""
        try:
            data = request.get_json() or {}
            rebuild = data.get('rebuild_model', False)
            
            trainer = ModelTrainer()
            
            if rebuild or not trainer.load_trained_model()[0]:
                logger.info("Starting model training...")
                success = trainer.train_model()
                
                return {
                    'success': success,
                    'message': '模型训练完成' if success else '模型训练失败',
                    'model_path': config.OUTPUT_MODEL_PATH if success else None
                }, 200 if success else 500
            else:
                return {
                    'success': True,
                    'message': '模型已存在，无需重新训练',
                    'model_path': config.OUTPUT_MODEL_PATH
                }, 200
                
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return {
                'success': False,
                'message': '训练启动失败',
                'error': str(e)
            }, 500

@training_ns.route('/status')
class TrainingStatus(Resource):
    @training_ns.doc('get_training_status', description='获取训练状态')
    def get(self):
        """获取训练状态"""
        try:
            trainer = ModelTrainer()
            model, tokenizer = trainer.load_trained_model()
            
            return {
                'model_exists': model is not None,
                'model_path': config.OUTPUT_MODEL_PATH,
                'training_data_path': config.TRAINING_DATA_PATH,
                'config': {
                    'model_name': config.MODEL_NAME,
                    'lora_rank': config.LORA_RANK,
                    'learning_rate': config.LEARNING_RATE,
                    'num_epochs': config.NUM_EPOCHS
                }
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {'error': str(e)}, 500

# Knowledge base endpoints
@knowledge_ns.route('/search')
class KnowledgeSearch(Resource):
    @knowledge_ns.expect(knowledge_request)
    @knowledge_ns.doc('search_knowledge', description='搜索知识库')
    def post(self):
        """搜索知识库"""
        try:
            if not agent_functions:
                return {'error': 'Agent not initialized'}, 500
            
            data = request.get_json()
            if not data or 'query' not in data:
                return {'error': 'Query is required'}, 400
            
            query = data['query']
            top_k = data.get('top_k', 5)
            
            result = agent_functions.search_knowledge_base(query, top_k)
            return result, 200
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return {'error': str(e)}, 500

@knowledge_ns.route('/stats')
class KnowledgeStats(Resource):
    @knowledge_ns.doc('get_knowledge_stats', description='获取知识库统计信息')
    def get(self):
        """获取知识库统计信息"""
        try:
            if not agent:
                return {'error': 'Agent not initialized'}, 500
            
            stats = agent.knowledge_base.get_stats()
            return stats, 200
            
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {'error': str(e)}, 500

@knowledge_ns.route('/rebuild')
class KnowledgeRebuild(Resource):
    @knowledge_ns.doc('rebuild_knowledge_index', description='重建知识库索引')
    def post(self):
        """重建知识库索引"""
        try:
            if not agent:
                return {'error': 'Agent not initialized'}, 500
            
            success = agent.knowledge_base.build_index(rebuild=True)
            return {
                'success': success,
                'message': '知识库索引重建完成' if success else '知识库索引重建失败'
            }, 200 if success else 500
            
        except Exception as e:
            logger.error(f"Error rebuilding knowledge index: {e}")
            return {'error': str(e)}, 500

# SQL endpoints
@sql_ns.route('/query')
class SQLQuery(Resource):
    @sql_ns.expect(sql_request)
    @sql_ns.doc('execute_sql_query', description='执行自然语言SQL查询')
    def post(self):
        """执行自然语言SQL查询"""
        try:
            if not agent_functions:
                return {'error': 'Agent not initialized'}, 500
            
            data = request.get_json()
            if not data or 'question' not in data:
                return {'error': 'Question is required'}, 400
            
            question = data['question']
            result = agent_functions.execute_database_query(question)
            return result, 200
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return {'error': str(e)}, 500

@sql_ns.route('/status')
class SQLStatus(Resource):
    @sql_ns.doc('get_sql_status', description='获取Text2SQL状态')
    def get(self):
        """获取Text2SQL状态"""
        try:
            if not agent:
                return {'error': 'Agent not initialized'}, 500
            
            status = agent.text2sql.get_status()
            return status, 200
            
        except Exception as e:
            logger.error(f"Error getting SQL status: {e}")
            return {'error': str(e)}, 500

@sql_ns.route('/samples')
class SQLSamples(Resource):
    @sql_ns.doc('get_sample_questions', description='获取示例问题')
    def get(self):
        """获取示例问题"""
        try:
            if not agent:
                return {'error': 'Agent not initialized'}, 500
            
            samples = agent.text2sql.get_sample_questions()
            return {'samples': samples}, 200
            
        except Exception as e:
            logger.error(f"Error getting sample questions: {e}")
            return {'error': str(e)}, 500

# System endpoints
@system_ns.route('/status')
class SystemStatus(Resource):
    @system_ns.doc('get_system_status', description='获取系统状态')
    def get(self):
        """获取系统状态"""
        try:
            if agent:
                status = agent.get_system_status()
                status['agent_initialized'] = True
            else:
                status = {
                    'agent_initialized': False,
                    'error': 'Agent not initialized'
                }
            
            return status, 200
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}, 500

@system_ns.route('/health')
class SystemHealth(Resource):
    @system_ns.doc('health_check', description='健康检查')
    def get(self):
        """健康检查"""
        return {
            'status': 'healthy',
            'service': 'AI Customer Service Agent',
            'version': '1.0.0',
            'agent_available': agent is not None
        }, 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal server error occurred'
    }), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': 'The request was malformed'
    }), 400

# Main route for testing
@app.route('/')
def home():
    return jsonify({
        'message': 'AI客户服务Agent API',
        'version': '1.0.0',
        'docs': '/docs/',
        'health': '/api/v1/system/health'
    })

if __name__ == '__main__':
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.FLASK_DEBUG
    )