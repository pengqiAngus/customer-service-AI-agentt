#!/usr/bin/env python3
"""
AI客户服务Agent快速启动脚本
"""

import sys
import os
import subprocess
import time
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """检查系统要求"""
    logger.info("🔍 检查系统要求...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ 是必需的")
        return False
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.warning("⚠️  .env 文件不存在，将使用默认配置")
    
    # Check if requirements are installed
    try:
        import flask
        import langchain
        import transformers
        logger.info("✅ 核心依赖已安装")
    except ImportError as e:
        logger.error(f"❌ 缺少依赖: {e}")
        logger.info("💡 请运行: pip install -r requirements.txt")
        return False
    
    return True

def install_dependencies():
    """安装依赖"""
    logger.info("📦 安装Python依赖...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        logger.info("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 依赖安装失败: {e}")
        return False

def check_services():
    """检查外部服务状态"""
    logger.info("🔍 检查外部服务...")
    
    services_status = {}
    
    # Check Redis
    try:
        import redis
        from app.config import config
        client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
        client.ping()
        services_status['redis'] = True
        logger.info("✅ Redis连接正常")
    except Exception as e:
        services_status['redis'] = False
        logger.warning(f"⚠️  Redis连接失败: {e}")
        logger.info("💡 将使用内存存储作为回退")
    
    # Check Database (if enabled)
    try:
        from app.config import config
        if config.ENABLE_TEXT2SQL and config.DATABASE_URL:
            import sqlalchemy
            engine = sqlalchemy.create_engine(config.DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            services_status['database'] = True
            logger.info("✅ 数据库连接正常")
        else:
            services_status['database'] = None
            logger.info("ℹ️  Text2SQL功能已禁用")
    except Exception as e:
        services_status['database'] = False
        logger.warning(f"⚠️  数据库连接失败: {e}")
        logger.info("💡 Text2SQL功能将不可用")
    
    return services_status

def train_model_if_needed():
    """如果需要，训练模型"""
    logger.info("🤖 检查模型状态...")
    
    from app.models.trainer import ModelTrainer
    trainer = ModelTrainer()
    
    model, tokenizer = trainer.load_trained_model()
    if model is None:
        logger.info("🔥 开始训练模型...")
        success = trainer.train_model()
        if success:
            logger.info("✅ 模型训练完成")
        else:
            logger.error("❌ 模型训练失败")
            return False
    else:
        logger.info("✅ 模型已存在，跳过训练")
    
    return True

def build_knowledge_base():
    """构建知识库"""
    logger.info("📚 初始化知识库...")
    
    from app.agent.rag import KnowledgeBase
    kb = KnowledgeBase()
    
    if kb.vector_store.count() == 0:
        logger.info("🔨 构建知识库索引...")
        success = kb.build_index()
        if success:
            logger.info("✅ 知识库索引构建完成")
        else:
            logger.error("❌ 知识库索引构建失败")
            return False
    else:
        logger.info("✅ 知识库索引已存在")
    
    return True

def start_application():
    """启动应用"""
    logger.info("🚀 启动AI客户服务Agent...")
    
    try:
        from app.main import main
        main()
    except Exception as e:
        logger.error(f"❌ 应用启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI客户服务Agent快速启动')
    parser.add_argument('--skip-checks', action='store_true', help='跳过系统检查')
    parser.add_argument('--install-deps', action='store_true', help='安装依赖')
    parser.add_argument('--train-model', action='store_true', help='强制训练模型')
    parser.add_argument('--rebuild-kb', action='store_true', help='重建知识库')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 AI客户服务Agent启动程序")
    print("=" * 60)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            sys.exit(1)
    
    # Check requirements
    if not args.skip_checks:
        if not check_requirements():
            sys.exit(1)
        
        # Check services
        services_status = check_services()
    
    # Train model if needed or requested
    if args.train_model:
        logger.info("🔄 强制重新训练模型...")
        from app.models.trainer import ModelTrainer
        trainer = ModelTrainer()
        success = trainer.train_model()
        if not success:
            logger.error("❌ 模型训练失败")
            sys.exit(1)
    else:
        if not train_model_if_needed():
            sys.exit(1)
    
    # Build knowledge base
    if args.rebuild_kb:
        logger.info("🔄 强制重建知识库...")
        from app.agent.rag import KnowledgeBase
        kb = KnowledgeBase()
        success = kb.build_index(rebuild=True)
        if not success:
            logger.error("❌ 知识库重建失败")
            sys.exit(1)
    else:
        if not build_knowledge_base():
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 系统初始化完成")
    print("🌐 访问 http://localhost:5000 开始使用")
    print("📚 API文档: http://localhost:5000/docs/")
    print("=" * 60)
    
    # Start application
    start_application()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 应用已停止")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        sys.exit(1)