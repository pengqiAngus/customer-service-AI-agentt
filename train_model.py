#!/usr/bin/env python3
"""
独立的模型训练脚本
运行此脚本来训练客户服务AI模型
"""

import sys
import os
import argparse
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from app.models.trainer import ModelTrainer
from app.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """训练模型主函数"""
    parser = argparse.ArgumentParser(description='训练AI客户服务模型')
    parser.add_argument(
        '--force-rebuild', 
        action='store_true', 
        help='强制重新训练模型，即使已存在训练好的模型'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("🤖 开始AI客户服务模型训练...")
        logger.info(f"📊 基础模型: {config.MODEL_NAME}")
        logger.info(f"📁 训练数据: {config.TRAINING_DATA_PATH}")
        logger.info(f"💾 输出路径: {config.OUTPUT_MODEL_PATH}")
        logger.info(f"🎯 LoRA参数: rank={config.LORA_RANK}, alpha={config.LORA_ALPHA}")
        logger.info(f"📈 训练配置: lr={config.LEARNING_RATE}, epochs={config.NUM_EPOCHS}, batch_size={config.BATCH_SIZE}")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Check if model already exists
        if not args.force_rebuild:
            model, tokenizer = trainer.load_trained_model()
            if model is not None:
                logger.info("✅ 已存在训练好的模型，跳过训练")
                logger.info("💡 使用 --force-rebuild 参数强制重新训练")
                return
        
        # Start training
        logger.info("🔥 开始训练...")
        success = trainer.train_model()
        
        if success:
            logger.info("🎉 训练完成！")
            logger.info(f"📁 模型已保存至: {config.OUTPUT_MODEL_PATH}")
            
            # Test loading the trained model
            model, tokenizer = trainer.load_trained_model()
            if model is not None:
                logger.info("✅ 模型加载测试成功")
            else:
                logger.warning("⚠️  模型加载测试失败")
        else:
            logger.error("❌ 训练失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️  训练被用户中断")
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()