# AI客户服务Agent - 快速启动指南

## 🚀 一键启动

### 方式一：使用快速启动脚本（推荐）

```bash
# 安装依赖并启动
python start.py --install-deps

# 或者分步执行
pip install -r requirements.txt
python start.py
```

### 方式二：手动启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，至少配置 OPENAI_API_KEY

# 3. 训练模型（可选）
python train_model.py

# 4. 启动服务
python app/main.py
```

## 📱 快速测试

启动后访问以下地址：

- **API文档**: http://localhost:5000/docs/
- **健康检查**: http://localhost:5000/api/v1/system/health
- **系统状态**: http://localhost:5000/api/v1/system/status

### 发送测试消息

```bash
curl -X POST http://localhost:5000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，我想了解退款政策",
    "session_id": "test123"
  }'
```

## ⚙️ 主要配置

### 必需配置
```env
# OpenAI API密钥（必需）
OPENAI_API_KEY=your_openai_api_key_here
```

### 可选配置
```env
# 禁用Text2SQL（如果没有数据库）
ENABLE_TEXT2SQL=False

# Redis配置（如果可用）
REDIS_HOST=localhost
REDIS_PORT=6379
```

## 🔧 功能开关

### 启动参数
```bash
# 安装依赖
python start.py --install-deps

# 强制重新训练模型
python start.py --train-model

# 重建知识库索引
python start.py --rebuild-kb

# 跳过系统检查
python start.py --skip-checks
```

### 独立脚本
```bash
# 仅训练模型
python train_model.py

# 强制重新训练
python train_model.py --force-rebuild
```

## 📚 更多信息

详细文档请参考：[README_Python.md](README_Python.md)

## 🆘 常见问题

**Q: 启动失败？**
```bash
# 检查Python版本（需要3.8+）
python --version

# 检查依赖安装
pip list | grep -E "(flask|langchain|transformers)"
```

**Q: Redis连接失败？**
- 系统会自动使用内存存储作为回退
- 安装Redis: `docker run -d -p 6379:6379 redis:latest`

**Q: 模型下载缓慢？**
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
python train_model.py
```

---

🎉 **开始体验AI客户服务Agent吧！**