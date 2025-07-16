# AI客户服务Agent - Python版本

基于LangChain和LangGraph的智能客户服务AI Agent系统，支持模型训练、RAG知识库、Text2SQL查询和完整的对话记忆功能。

## 🚀 功能特性

### 第一步：模型训练微调
- 🔥 **LoRA微调**: 使用LoRA技术对预训练模型进行微调
- 📊 **自定义数据集**: 支持自定义客户服务对话数据集
- ⚡ **高效训练**: 使用Transformers和PEFT库进行高效训练
- 💾 **模型保存**: 自动保存和加载训练好的模型

### 第二步：AI Agent开发
- 🧠 **记忆功能**: 基于Redis的对话上下文记忆
- 📚 **RAG集成**: 知识库文档检索增强生成
- 🔍 **Text2SQL**: 自然语言转SQL查询功能
- 🔧 **Function Calling**: 集成多种功能调用
- 🌊 **LangGraph工作流**: 智能流程控制和状态管理

### Web API接口
- 🌐 **Flask框架**: RESTful API接口
- 📖 **Swagger文档**: 自动生成API文档和交互界面
- 🔄 **实时交互**: 支持会话管理和实时对话

## 📁 项目结构

```
AI-customer-service-agent-python/
├── app/
│   ├── config.py              # 配置管理
│   ├── main.py               # 主应用入口
│   ├── models/
│   │   └── trainer.py        # 模型训练器
│   ├── agent/
│   │   ├── memory.py         # 对话记忆
│   │   ├── rag.py           # RAG知识库
│   │   ├── text2sql.py      # Text2SQL功能
│   │   └── workflow.py      # LangGraph工作流
│   └── api/
│       └── routes.py        # Flask API路由
├── data/
│   ├── training_dataset.json    # 训练数据集
│   ├── knowledge_files/         # 知识库文件
│   ├── vector_store/           # 向量存储
│   └── database_schema.json   # 数据库模式
├── models/                     # 训练后的模型
├── requirements.txt           # Python依赖
├── .env                      # 环境变量配置
├── train_model.py           # 独立训练脚本
└── README_Python.md        # 项目说明
```

## 🛠️ 安装和配置

### 1. 环境要求
- Python 3.8+
- CUDA (可选，用于GPU加速)
- Redis (用于记忆功能)
- PostgreSQL (用于Text2SQL功能)

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd AI-customer-service-agent-python

# 安装Python依赖
pip install -r requirements.txt
# 或使用pnpm等包管理器
# pnpm install
```

### 3. 环境配置

复制并修改环境变量文件：

```bash
cp .env.example .env
```

编辑`.env`文件，配置以下关键参数：

```env
# OpenAI API配置 (必需)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 数据库配置 (Text2SQL功能)
DATABASE_URL=postgresql://user:password@localhost:5432/database
ENABLE_TEXT2SQL=True

# Redis配置 (记忆功能)
REDIS_HOST=localhost
REDIS_PORT=6379

# 模型训练配置
MODEL_NAME=microsoft/DialoGPT-medium
TRAINING_DATA_PATH=./data/training_dataset.json
OUTPUT_MODEL_PATH=./models/fine_tuned_model
```

### 4. 准备数据

系统会自动创建示例数据文件，你也可以准备自己的数据：

#### 训练数据格式 (`data/training_dataset.json`)
```json
[
  {
    "input": "如何申请退款？",
    "output": "您好！申请退款很简单，请提供您的订单号..."
  }
]
```

#### 知识库文件 (`data/knowledge_files/`)
- 支持格式: `.txt`, `.md`, `.pdf`, `.csv`, `.json`
- 系统会自动创建示例FAQ和政策文件

## 🚀 使用指南

### 第一步：训练模型

```bash
# 训练新模型
python train_model.py

# 强制重新训练
python train_model.py --force-rebuild
```

训练过程包括：
1. 加载预训练模型和数据集
2. 配置LoRA参数
3. 执行微调训练
4. 保存训练后的模型

### 第二步：启动AI Agent服务

```bash
# 启动Flask应用
python app/main.py

# 或使用模块方式运行
python -m app.main
```

服务启动后：
- 🌐 **API服务**: http://localhost:5000
- 📚 **Swagger文档**: http://localhost:5000/docs/
- 🏥 **健康检查**: http://localhost:5000/api/v1/system/health

## 📡 API接口说明

### 聊天对话接口

```bash
# 发送消息
curl -X POST http://localhost:5000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "如何申请退款？",
    "session_id": "user123"
  }'

# 获取会话信息
curl http://localhost:5000/api/v1/chat/session/user123

# 清除会话
curl -X DELETE http://localhost:5000/api/v1/chat/session/user123
```

### 模型训练接口

```bash
# 开始训练
curl -X POST http://localhost:5000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"rebuild_model": false}'

# 获取训练状态
curl http://localhost:5000/api/v1/training/status
```

### 知识库接口

```bash
# 搜索知识库
curl -X POST http://localhost:5000/api/v1/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "退款政策",
    "top_k": 5
  }'

# 重建索引
curl -X POST http://localhost:5000/api/v1/knowledge/rebuild
```

### Text2SQL接口

```bash
# 执行SQL查询
curl -X POST http://localhost:5000/api/v1/sql/query \
  -H "Content-Type: application/json" \
  -d '{"question": "查询所有客户信息"}'

# 获取示例问题
curl http://localhost:5000/api/v1/sql/samples
```

### 系统状态接口

```bash
# 系统状态
curl http://localhost:5000/api/v1/system/status

# 健康检查
curl http://localhost:5000/api/v1/system/health
```

## 🔧 核心组件说明

### 1. 记忆管理 (Memory)
- **Redis存储**: 对话历史持久化存储
- **上下文管理**: 自动管理对话上下文
- **内存回退**: Redis不可用时使用内存存储

### 2. RAG知识库 (RAG)
- **向量化存储**: 使用ChromaDB进行向量存储
- **文档加载**: 支持多种格式文档自动加载
- **语义搜索**: 基于语义相似度的文档检索

### 3. Text2SQL引擎
- **自然语言理解**: 将用户问题转换为SQL查询
- **安全验证**: 防止危险SQL操作
- **结果格式化**: 自动格式化查询结果

### 4. LangGraph工作流
- **状态管理**: 完整的对话状态跟踪
- **智能路由**: 根据意图自动选择处理路径
- **组件集成**: 无缝集成所有功能组件

## 🎯 高级配置

### LoRA训练参数调优

```env
# LoRA配置
LORA_RANK=16          # LoRA矩阵维度，较大值提高表达能力
LORA_ALPHA=32         # LoRA缩放参数
LEARNING_RATE=5e-5    # 学习率
NUM_EPOCHS=3          # 训练轮数
BATCH_SIZE=4          # 批次大小
```

### 知识库优化

```env
# RAG配置
CHUNK_SIZE=1000       # 文档分块大小
CHUNK_OVERLAP=200     # 分块重叠长度
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Text2SQL开关

```env
# 禁用Text2SQL功能 (如果没有数据库)
ENABLE_TEXT2SQL=False
```

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查模型路径和权限
   ls -la ./models/fine_tuned_model/
   ```

2. **Redis连接失败**
   ```bash
   # 启动Redis服务
   redis-server
   # 或使用Docker
   docker run -d -p 6379:6379 redis:latest
   ```

3. **数据库连接失败**
   ```bash
   # 检查PostgreSQL服务状态
   systemctl status postgresql
   # 测试连接
   psql postgresql://user:password@localhost:5432/database
   ```

4. **GPU内存不足**
   ```env
   # 减少批次大小
   BATCH_SIZE=2
   # 或使用CPU训练
   CUDA_VISIBLE_DEVICES=""
   ```

### 日志调试

```bash
# 查看应用日志
tail -f app.log

# 启用详细日志
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
python app/main.py
```

## 📊 性能优化

### 1. 模型优化
- 使用较小的基础模型减少内存占用
- 调整LoRA参数平衡性能和质量
- 使用量化技术减少模型大小

### 2. 检索优化
- 定期重建向量索引
- 调整文档分块策略
- 使用更高效的嵌入模型

### 3. 数据库优化
- 为常用查询字段建立索引
- 优化SQL查询语句
- 使用连接池管理数据库连接

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/AmazingFeature`
3. 提交更改: `git commit -m 'feat: add amazing feature'`
4. 推送分支: `git push origin feature/AmazingFeature`
5. 创建Pull Request

## 📄 许可证

本项目基于MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 强大的LLM应用框架
- [LangGraph](https://github.com/langchain-ai/langgraph) - 智能工作流引擎  
- [Transformers](https://github.com/huggingface/transformers) - 预训练模型库
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [Flask-RESTX](https://github.com/python-restx/flask-restx) - API文档生成

---

**🎉 开始构建您的智能客户服务系统吧！**