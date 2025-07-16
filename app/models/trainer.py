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
    """使用LoRA微调的客户服务模型训练器"""
    
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self.training_data_path = config.TRAINING_DATA_PATH
        self.output_path = config.OUTPUT_MODEL_PATH
        self.tokenizer = None
        self.model = None
        
    def load_training_data(self) -> Dataset:
        """加载和准备训练数据，支持目录下多文件合并"""
        try:
            data = []
            if os.path.isdir(self.training_data_path):
                # 遍历目录下所有包含 train 的 json 文件
                for fname in os.listdir(self.training_data_path):
                    print("fname",fname)
                    if 'train' in fname and fname.endswith('.json'):
                        file_path = os.path.join(self.training_data_path, fname)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    item = json.loads(line)
                                    # 只处理包含 content 和 summary 的行
                                    if 'content' in item and 'summary' in item:
                                        data.append({
                                            'input': item['content'],
                                            'output': item['summary']
                                        })
                                except Exception as e:
                                    logger.warning(f"跳过无效行: {line}，错误: {e}")
                if not data:
                    raise ValueError("未找到有效的训练数据文件或内容为空")
            else:
                # 兼容原有单文件 json 格式
                if not os.path.exists(self.training_data_path):
                    sample_data = self._create_sample_data()
                    os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
                    with open(self.training_data_path, 'w', encoding='utf-8') as f:
                        json.dump(sample_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"已创建示例训练数据: {self.training_data_path}")
                with open(self.training_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 兼容 content/summary 格式
                    if data and 'content' in data[0] and 'summary' in data[0]:
                        data = [
                            {'input': item['content'], 'output': item['summary']} for item in data
                        ]
            # 转换为数据集格式
            dataset = Dataset.from_list(data)
            return dataset
        except Exception as e:
            logger.error(f"加载训练数据出错: {e}")
            raise
    
    def _create_sample_data(self) -> List[Dict[str, str]]:
        """创建客户服务的示例训练数据"""
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
        """准备训练用的分词器和模型，自动适配Qwen3等模型的LoRA target_modules"""
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            # 自动适配LoRA target_modules
            # Qwen3结构参考官方文档和源码
            # https://github.com/QwenLM/Qwen3/blob/main/qwen3/modeling_qwen3.py
            # 主要为q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
            qwen3_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            # 检查模型中实际存在的模块
            model_named_modules = dict(self.model.named_modules())
            available_modules = set(model_named_modules.keys())
            lora_targets = [m for m in qwen3_modules if any(m in name for name in available_modules)]
            if not lora_targets:
                # fallback: 尝试所有qwen3推荐模块
                lora_targets = qwen3_modules
            logger.info(f"LoRA target_modules: {lora_targets}")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.LORA_RANK,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=0.1,
                target_modules=lora_targets
            )
            # 应用LoRA到模型
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logger.info("模型和分词器加载成功")
        except Exception as e:
            logger.error(f"准备模型出错: {e}")
            raise
    
    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """预处理训练用数据集"""
        def tokenize_function(examples):
            # 格式化输入输出对
            texts = []
            for inp, out in zip(examples['input'], examples['output']):
                text = f"用户: {inp}\n客服: {out}{self.tokenizer.eos_token}"
                texts.append(text)
            
            # 分词
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 设置语言建模标签
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # 应用分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_model(self) -> bool:
        """主训练函数"""
        try:
            logger.info("Starting model training...")
            
            # 加载数据
            dataset = self.load_training_data()
            logger.info(f"Loaded {len(dataset)} training examples")
            
            # 准备模型和分词器
            self.prepare_tokenizer_and_model()
            
            # 预处理数据
            tokenized_dataset = self.preprocess_data(dataset)
            
            # 划分数据集（80%训练，20%验证）
            split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            
            # 训练参数
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
            
            # 数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # 创建Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # 开始训练
            logger.info("Training started...")
            trainer.train()
            
            # 保存模型
            os.makedirs(self.output_path, exist_ok=True)
            trainer.save_model(self.output_path)
            self.tokenizer.save_pretrained(self.output_path)
            
            logger.info(f"训练完成！模型已保存到: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return False
    
    def load_trained_model(self):
        """加载已训练好的模型用于推理"""
        try:
            if not os.path.exists(self.output_path):
                logger.warning(f"未找到已训练模型: {self.output_path}")
                return None, None
            
            tokenizer = AutoTokenizer.from_pretrained(self.output_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.output_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info("已成功加载训练好的模型")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"加载训练模型出错: {e}")
            return None, None

# 训练脚本入口
if __name__ == "__main__":
    trainer = ModelTrainer()
    success = trainer.train_model()
    
    if success:
        print("✅ 模型训练完成！")
    else:
        print("❌ 模型训练失败！")