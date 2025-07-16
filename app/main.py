#!/usr/bin/env python3
"""
AI客户服务Agent主应用入口
"""

import sys
import os
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.api.routes import app
from app.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    try:
        logger.info("🚀 启动AI客户服务Agent系统...")
        logger.info(f"📍 服务地址: http://{config.HOST}:{config.PORT}")
        logger.info(f"📚 API文档: http://{config.HOST}:{config.PORT}/docs/")
        logger.info(f"🔧 环境: {config.FLASK_ENV}")
        
        # Start Flask application
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.FLASK_DEBUG,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 应用已停止")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()