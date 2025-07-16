import json
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Dict, List, Any
import logging

from app.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model trainer using LoRA fine-tuning for customer service responses"""
    
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self.training_data_path = config.TRAINING_DATA_PATH
        self.output_path = config.OUTPUT_MODEL_PATH
        self.tokenizer = None
        self.model = None
        
    def load_training_data(self) -> Dataset:
        """Load and prepare training data"""
        try:
            if not os.path.exists(self.training_data_path):
                # Create sample training data if file doesn't exist
                sample_data = self._create_sample_data()
                os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
                with open(self.training_data_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Created sample training data at {self.training_data_path}")
            
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to dataset format
            dataset = Dataset.from_list(data)
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def _create_sample_data(self) -> List[Dict[str, str]]:
        """Create sample training data for customer service"""
        return [
            {
                "input": "如何申请退款？",
                "output": "您好！申请退款很简单，请提供您的订单号，我会为您立即处理退款申请。通常退款会在3-5个工作日内到账。"
            },
            {
                "input": "订单状态怎么查询？",
                "output": "您可以通过以下方式查询订单状态：1) 登录我们的官网账户中心；2) 使用订单号在首页查询；3) 联系客服提供订单号查询。"
            },
            {
                "input": "产品质量有问题怎么办？",
                "output": "非常抱歉给您带来不便！如果产品质量有问题，我们提供：1) 7天无理由退货；2) 免费换货服务；3) 质量问题全额退款。请提供订单号和问题照片。"
            },
            {
                "input": "配送时间需要多久？",
                "output": "我们的配送时间如下：1) 同城24小时内送达；2) 省内2-3天送达；3) 跨省3-5天送达。急件可选择加急配送服务。"
            },
            {
                "input": "客服工作时间是？",
                "output": "我们的客服工作时间是：周一至周日 9:00-22:00。非工作时间您可以留言，我们会在工作时间内第一时间回复您。"
            }
        ]
    
    def prepare_tokenizer_and_model(self):
        """Prepare tokenizer and model for training"""
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.LORA_RANK,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"]  # For GPT-2 style models
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error preparing model: {e}")
            raise
    
    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset for training"""
        def tokenize_function(examples):
            # Format input-output pairs
            texts = []
            for inp, out in zip(examples['input'], examples['output']):
                text = f"用户: {inp}\n客服: {out}{self.tokenizer.eos_token}"
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Set labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_model(self) -> bool:
        """Main training function"""
        try:
            logger.info("Starting model training...")
            
            # Load data
            dataset = self.load_training_data()
            logger.info(f"Loaded {len(dataset)} training examples")
            
            # Prepare model and tokenizer
            self.prepare_tokenizer_and_model()
            
            # Preprocess data
            tokenized_dataset = self.preprocess_data(dataset)
            
            # Split dataset (80% train, 20% eval)
            split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.output_path,
                num_train_epochs=config.NUM_EPOCHS,
                per_device_train_batch_size=config.BATCH_SIZE,
                per_device_eval_batch_size=config.BATCH_SIZE,
                warmup_steps=100,
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=50,
                save_steps=100,
                learning_rate=config.LEARNING_RATE,
                fp16=torch.cuda.is_available(),
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Start training
            logger.info("Training started...")
            trainer.train()
            
            # Save model
            os.makedirs(self.output_path, exist_ok=True)
            trainer.save_model(self.output_path)
            self.tokenizer.save_pretrained(self.output_path)
            
            logger.info(f"Training completed! Model saved to {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def load_trained_model(self):
        """Load the trained model for inference"""
        try:
            if not os.path.exists(self.output_path):
                logger.warning(f"Trained model not found at {self.output_path}")
                return None, None
            
            tokenizer = AutoTokenizer.from_pretrained(self.output_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.output_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info("Trained model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            return None, None

# Training script entry point
if __name__ == "__main__":
    trainer = ModelTrainer()
    success = trainer.train_model()
    
    if success:
        print("✅ 模型训练完成！")
    else:
        print("❌ 模型训练失败！")