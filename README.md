# AI客户助手系统

基于 `langchain.js` 和 `NestJS` 开发的智能客户助手系统，集成了模型微调、RAG知识库、对话记忆和Text2SQL等功能。

## 🚀 功能特性

### 第一步：模型训练微调
- ✅ **LoRA微调训练**：使用LoRA技术进行高效模型微调
- ✅ **训练数据管理**：支持从URL加载训练数据集
- ✅ **训练监控**：实时监控训练进度和损失
- ✅ **模型验证**：训练完成后自动验证模型效果

### 第二步：AI Agent功能
- ✅ **对话记忆（Memory）**：使用langchain记忆系统保持对话上下文
- ✅ **RAG知识库**：集成向量数据库，支持文档检索和相似性搜索
- ✅ **Text2SQL功能**：自然语言转SQL查询，支持数据库查询
- ✅ **Function Calling**：智能函数调用，集成所有功能模块

## 🏗️ 技术架构

```
├── 训练模块 (Training Module)
│   ├── 模型微调 (LoRA Fine-tuning)
│   ├── 数据预处理 (Data Preprocessing)
│   └── 训练监控 (Training Monitoring)
│
├── AI Agent模块 (Agent Module)
│   ├── 核心服务 (Core Service)
│   ├── 对话记忆 (Conversation Memory)
│   ├── RAG知识库 (Knowledge Base)
│   ├── Text2SQL引擎 (Text2SQL Engine)
│   └── Function Calling (Function Calling)
│
└── API接口层 (Controller Layer)
    ├── 训练管理接口
    ├── 对话交互接口
    ├── 知识库管理接口
    └── 数据查询接口
```

## 📦 安装与运行

### 1. 安装依赖

```bash
# 使用pnpm安装依赖
pnpm install
```

### 2. 环境配置

复制环境变量模板并配置：

```bash
cp .env.example .env
```

配置 `.env` 文件：

```env
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# 数据库配置
DATABASE_URL=sqlite:./data/agent.db

# 模型训练配置
TRAINING_DATA_URL=https://example.com/api/training-data
LORA_MODEL_PATH=./models/lora-fine-tuned
BASE_MODEL_URL=https://huggingface.co/microsoft/DialoGPT-medium

# RAG配置
KNOWLEDGE_BASE_URL=https://example.com/api/knowledge-files
VECTOR_STORE_PATH=./data/vector-store
EMBEDDING_MODEL=text-embedding-ada-002

# 服务配置
PORT=3000
NODE_ENV=development
```

### 3. 构建项目

```bash
pnpm run build
```

### 4. 启动服务

```bash
# 开发模式
pnpm run start:dev

# 生产模式
pnpm run start:prod
```

### 5. 访问服务

- **API文档**: http://localhost:3000/docs
- **主要接口**: http://localhost:3000/api

## 🎯 API使用指南

### 模型训练相关

#### 开始训练
```bash
POST /api/training/start
```

#### 获取模型状态
```bash
GET /api/training/status/{modelPath}
```

#### 验证模型
```bash
POST /api/training/validate/{modelPath}
```

### AI Agent对话

#### 与AI助手对话
```bash
POST /api/agent/chat
Content-Type: application/json

{
  "message": "你好，我想查询我的订单状态",
  "sessionId": "session_123",
  "userId": "user_456"
}
```

#### 获取会话分析
```bash
GET /api/agent/session/{sessionId}/analysis
```

#### 清除会话记忆
```bash
DELETE /api/agent/session/{sessionId}
```

### 知识库管理

#### 添加知识文档
```bash
POST /api/agent/knowledge/documents
Content-Type: application/json

{
  "title": "产品使用指南",
  "content": "详细的产品使用说明...",
  "category": "产品文档",
  "metadata": {
    "version": "1.0",
    "author": "技术团队"
  }
}
```

#### 搜索知识库
```bash
POST /api/agent/knowledge/search
Content-Type: application/json

{
  "query": "如何重置密码",
  "limit": 5
}
```

### Text2SQL查询

#### 自然语言查询
```bash
POST /api/agent/sql/query
Content-Type: application/json

{
  "question": "查询所有未完成的订单"
}
```

#### 获取数据库结构
```bash
GET /api/agent/sql/schemas
```

## 🔧 脚本命令

```bash
# 模型训练
pnpm run train:model

# 数据库初始化
pnpm run setup:db

# 代码格式化
pnpm run format

# 运行测试
pnpm run test

# 生成覆盖率报告
pnpm run test:cov
```

## 📁 项目结构

```
customer-service-ai-agent/
├── src/                          # 源代码目录
│   ├── training/                 # 模型训练模块
│   │   ├── model-trainer.ts      # LoRA微调训练
│   │   ├── training.controller.ts # 训练接口控制器
│   │   └── training.module.ts    # 训练模块定义
│   │
│   ├── agent/                    # AI Agent模块
│   │   ├── memory/               # 对话记忆
│   │   │   └── conversation-memory.ts
│   │   ├── rag/                  # RAG知识库
│   │   │   └── knowledge-base.ts
│   │   ├── sql/                  # Text2SQL
│   │   │   └── text2sql.ts
│   │   ├── ai-agent.service.ts   # AI Agent核心服务
│   │   ├── agent.controller.ts   # Agent接口控制器
│   │   └── agent.module.ts       # Agent模块定义
│   │
│   ├── app.module.ts             # 主应用模块
│   └── main.ts                   # 应用入口
│
├── data/                         # 数据目录
│   ├── conversations/            # 对话历史
│   ├── knowledge/                # 知识库文档
│   ├── training/                 # 训练数据
│   └── vector-store/             # 向量存储
│
├── models/                       # 模型目录
│   └── lora-fine-tuned/          # LoRA微调模型
│
├── package.json                  # 项目配置
├── tsconfig.json                 # TypeScript配置
├── nest-cli.json                 # NestJS CLI配置
└── .env.example                  # 环境变量模板
```

## 🌟 核心功能详解

### 1. LoRA模型微调

使用低秩适应(LoRA)技术进行高效微调：

```typescript
const loraConfig = {
  rank: 16,           // LoRA的秩
  alpha: 32,          // LoRA的缩放参数
  dropout: 0.1,       // Dropout率
  targetModules: [    // 目标模块
    'q_proj', 'k_proj', 'v_proj', 'o_proj'
  ],
  taskType: 'CAUSAL_LM'
};
```

### 2. 对话记忆系统

使用langchain的ConversationSummaryBufferMemory：

```typescript
const memory = new ConversationSummaryBufferMemory({
  llm: this.llm,
  maxTokenLimit: 100,
  returnMessages: true,
});
```

### 3. RAG知识库

基于FAISS向量数据库的相似性搜索：

```typescript
const searchResults = await this.vectorStore.similaritySearchWithScore(query, 5);
```

### 4. Text2SQL引擎

自然语言到SQL的智能转换：

```typescript
const sqlQuery = await this.generateSQL(question);
const results = await this.executeSQL(sqlQuery);
```

## 🛡️ 安全特性

- ✅ SQL注入防护
- ✅ 输入验证和清理
- ✅ 环境变量管理
- ✅ CORS配置
- ✅ 错误处理和日志记录

## 📈 监控和调试

### 健康检查
```bash
GET /api/agent/health
```

### 服务统计
```bash
GET /api/agent/stats
```

### 日志级别
- INFO: 一般信息
- WARN: 警告信息
- ERROR: 错误信息
- DEBUG: 调试信息

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain.js](https://js.langchain.com/) - AI应用开发框架
- [NestJS](https://nestjs.com/) - Node.js服务端框架
- [OpenAI](https://openai.com/) - AI模型和API服务
- [FAISS](https://github.com/facebookresearch/faiss) - 向量相似性搜索库

---

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

- 📧 Email: support@company.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)