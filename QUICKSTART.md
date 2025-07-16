# 🚀 快速启动指南

## 📋 前置要求

- Node.js (v18+)
- pnpm 包管理器
- OpenAI API Key

## ⚡ 5分钟快速启动

### 1. 安装依赖
```bash
pnpm install
```

### 2. 配置环境变量
```bash
# 编辑 .env 文件，设置你的 OpenAI API Key
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. 构建项目
```bash
pnpm run build
```

### 4. 启动服务
```bash
pnpm run start:dev
```

### 5. 验证服务
访问 http://localhost:3000/docs 查看API文档

### 6. 运行演示
打开新终端窗口：
```bash
node scripts/demo.js
```

## 🎯 核心功能测试

### 模型训练
```bash
curl -X POST http://localhost:3000/api/training/start
```

### AI对话
```bash
curl -X POST http://localhost:3000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，我想了解产品保修政策",
    "sessionId": "test_session_123"
  }'
```

### 知识库搜索
```bash
curl -X POST http://localhost:3000/api/agent/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "保修",
    "limit": 3
  }'
```

### Text2SQL查询
```bash
curl -X POST http://localhost:3000/api/agent/sql/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "查询所有用户信息"
  }'
```

## 🛠️ 常见问题

### Q: OpenAI API 调用失败？
A: 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确设置

### Q: 数据库错误？
A: 确保 `./data` 目录有写入权限，系统会自动创建SQLite数据库

### Q: 向量存储初始化失败？
A: 检查 faiss-node 依赖是否正确安装，可能需要重新安装

### Q: 模型训练功能不工作？
A: 这是演示环境，实际训练需要配置真实的训练数据和模型地址

## 📖 更多信息

- 详细文档：[README.md](./README.md)
- API文档：http://localhost:3000/docs
- 演示脚本：`node scripts/demo.js`