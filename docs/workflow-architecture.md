# LangGraph工作流架构

## 🏗️ 概述

我们将原来的if-else意图识别逻辑重构为基于LangGraph的智能工作流，实现了更优雅、可扩展的状态机模式。

## 🔄 工作流设计

### 状态图结构

```
┌─────────────────┐
│  intent_analysis │ ← 入口点
└─────────┬───────┘
          │
          ▼ (条件路由)
┌─────────────────┐
│   routeBasedOn   │
│     Intent       │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐ ┌─────────┐
│sql_proc.│ │knowledge│
│essing   │ │processing│
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
┌─────────────────┐
│general_convers. │
└────┬────┬──────┘
     │    │
     ▼    ▼
┌─────────────────┐
│response_generat.│
└─────────┬───────┘
          │
          ▼
         END
```

### 节点说明

#### 1. intent_analysis (意图分析)
- **功能**: 分析用户消息意图
- **输入**: 用户消息、对话历史
- **输出**: 意图分类、置信度、路由决策
- **AI模型**: GPT-3.5-turbo

#### 2. sql_processing (SQL处理)
- **功能**: 自然语言转SQL查询
- **输入**: 用户消息
- **输出**: SQL查询结果、解释
- **工具**: Text2SQL引擎

#### 3. knowledge_processing (知识库处理)
- **功能**: RAG知识库搜索
- **输入**: 用户消息
- **输出**: 相关文档、来源
- **工具**: FAISS向量搜索

#### 4. general_conversation (常规对话)
- **功能**: 基于记忆的对话
- **输入**: 用户消息、对话历史
- **输出**: 上下文相关回复
- **工具**: 对话记忆系统

#### 5. response_generation (响应生成)
- **功能**: 生成最终回复
- **输入**: 处理结果、上下文
- **输出**: 格式化回复
- **AI模型**: GPT-3.5-turbo

## 🛤️ 路由逻辑

### 条件边函数
```typescript
private routeBasedOnIntent(state: WorkflowState): string {
  if (state.needsSQL) {
    return 'sql';
  } else if (state.needsKnowledge) {
    return 'knowledge';
  } else {
    return 'general';
  }
}
```

### 路由规则
1. **SQL查询**: 用户询问具体数据、统计信息、订单查询等
2. **知识库查询**: 用户询问政策、流程、产品信息、技术支持等
3. **常规对话**: 普通聊天、问候、感谢等

## 📊 状态管理

### WorkflowState接口
```typescript
interface WorkflowState {
  message: string;
  sessionId: string;
  userId?: string;
  intent: string;
  needsSQL: boolean;
  needsKnowledge: boolean;
  confidence: number;
  conversationHistory: string;
  response?: string;
  sources?: string[];
  sqlResults?: SQLQueryResult;
  searchResults?: SearchResult[];
  executionTime: number;
  metadata: {
    usedMemory: boolean;
    usedRAG: boolean;
    usedSQL: boolean;
    workflowPath: string[];
  };
}
```

### 状态流转
1. **初始化**: 构建初始状态，包含用户消息和会话信息
2. **意图分析**: 更新意图和路由决策
3. **分支处理**: 根据意图执行相应的处理节点
4. **响应生成**: 基于处理结果生成最终回复
5. **状态完成**: 记录执行路径和元数据

## 🎯 优势对比

### 原来的if-else方式
```typescript
// ❌ 硬编码的if-else逻辑
if (intent.needsSQL) {
  // SQL处理
} else if (intent.needsKnowledge) {
  // 知识库处理
} else {
  // 常规对话
}
```

### 现在的LangGraph方式
```typescript
// ✅ 声明式的工作流定义
workflow.addConditionalEdges(
  'intent_analysis',
  routeBasedOnIntent,
  {
    sql: 'sql_processing',
    knowledge: 'knowledge_processing',
    general: 'general_conversation'
  }
);
```

## 🚀 扩展性

### 添加新节点
```typescript
// 1. 添加新节点
workflow.addNode('new_processing', this.newProcessing.bind(this));

// 2. 添加路由
workflow.addConditionalEdges(
  'intent_analysis',
  routeBasedOnIntent,
  {
    sql: 'sql_processing',
    knowledge: 'knowledge_processing',
    general: 'general_conversation',
    new: 'new_processing'  // 新增路由
  }
);

// 3. 连接后续节点
workflow.addEdge('new_processing', 'response_generation');
```

### 添加新状态
```typescript
// 在WorkflowState接口中添加新字段
interface WorkflowState {
  // ... 现有字段
  newField?: string;
  newResults?: any;
}
```

## 📈 监控和调试

### 执行路径追踪
```typescript
// 每个节点都会记录执行路径
metadata: {
  workflowPath: ['intent_analysis', 'sql_processing', 'response_generation']
}
```

### 性能监控
```typescript
// 记录每个节点的执行时间
executionTime: number;
```

### 状态可视化
```typescript
// 获取工作流统计信息
getWorkflowStats(): {
  nodes: string[];
  edges: string[];
  conditionalEdges: string[];
  description: string;
}
```

## 🔧 配置和调优

### 意图分析提示词
```typescript
const prompt = PromptTemplate.fromTemplate(`
分析用户消息的意图，判断需要使用哪种处理方式。

用户消息: {message}
对话历史: {history}

分析标准：
1. SQL查询：用户询问具体数据、统计信息、订单查询、用户信息等
2. 知识库查询：用户询问政策、流程、产品信息、技术支持等
3. 常规对话：普通聊天、问候、感谢等

请以JSON格式返回:
{
  "intent": "意图描述",
  "needsSQL": true/false,
  "needsKnowledge": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "分析理由"
}
`);
```

### 置信度阈值
```typescript
// 可以根据置信度调整路由策略
if (state.confidence < 0.7) {
  // 低置信度时使用更保守的策略
  return 'general';
}
```

## 🎉 总结

通过使用LangGraph重构意图识别逻辑，我们实现了：

1. **更好的可维护性**: 声明式的工作流定义
2. **更强的扩展性**: 易于添加新节点和路由
3. **更清晰的逻辑**: 状态机模式比if-else更直观
4. **更好的监控**: 完整的执行路径追踪
5. **更强的调试能力**: 详细的状态信息和元数据

这种架构为AI Agent的未来扩展提供了坚实的基础。 