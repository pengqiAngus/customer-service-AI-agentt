import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { LLMChain } from 'langchain/chains';
import { AgentExecutor, createReactAgent } from 'langchain/agents';
import { Tool } from '@langchain/core/tools';
import { ConversationMemory } from './memory/conversation-memory';
import { KnowledgeBase, SearchResult } from './rag/knowledge-base';
import { Text2SQL, SQLQueryResult } from './sql/text2sql';

export interface ChatMessage {
  message: string;
  sessionId: string;
  userId?: string;
}

export interface ChatResponse {
  response: string;
  sessionId: string;
  timestamp: Date;
  sources?: string[];
  sqlResults?: SQLQueryResult;
  metadata: {
    usedMemory: boolean;
    usedRAG: boolean;
    usedSQL: boolean;
    executionTime: number;
  };
}

@Injectable()
export class AIAgentService {
  private readonly logger = new Logger(AIAgentService.name);
  private llm: ChatOpenAI;
  private agent: AgentExecutor;
  private tools: Tool[];

  constructor(
    private configService: ConfigService,
    private conversationMemory: ConversationMemory,
    private knowledgeBase: KnowledgeBase,
    private text2sql: Text2SQL,
  ) {
    this.initializeAgent();
  }

  /**
   * 初始化AI Agent
   */
  private async initializeAgent(): Promise<void> {
    this.logger.log('初始化AI Agent...');

    // 初始化LLM
    this.llm = new ChatOpenAI({
      openAIApiKey: this.configService.get<string>('OPENAI_API_KEY'),
      modelName: this.configService.get<string>('OPENAI_MODEL') || 'gpt-3.5-turbo',
      temperature: 0.7,
    });

    // 创建工具
    this.tools = await this.createTools();

    // 创建Agent
    const prompt = this.createAgentPrompt();
    this.agent = await createReactAgent({
      llm: this.llm,
      tools: this.tools,
      prompt: prompt,
    });

    this.logger.log('AI Agent 初始化完成');
  }

  /**
   * 处理聊天消息
   */
  async chat(chatMessage: ChatMessage): Promise<ChatResponse> {
    const startTime = Date.now();
    const { message, sessionId, userId } = chatMessage;

    this.logger.log(`处理用户消息: ${message} (会话: ${sessionId})`);

    try {
      // 1. 记录用户消息到记忆
      await this.conversationMemory.addMessage(sessionId, 'user', message, { userId });

      // 2. 获取会话历史
      const memoryVariables = await this.conversationMemory.getMemoryVariables(sessionId);
      
      // 3. 分析用户意图
      const intent = await this.analyzeUserIntent(message);
      
      // 4. 根据意图选择处理策略
      let response: string;
      let sources: string[] = [];
      let sqlResults: SQLQueryResult | undefined;
      let usedMemory = false;
      let usedRAG = false;
      let usedSQL = false;

      if (intent.needsSQL) {
        // 需要查询数据库
        this.logger.log('检测到需要SQL查询');
        sqlResults = await this.text2sql.naturalLanguageToSQL(message);
        response = await this.generateResponseWithSQL(message, sqlResults, memoryVariables.history);
        usedSQL = true;
      } else if (intent.needsKnowledge) {
        // 需要搜索知识库
        this.logger.log('检测到需要知识库查询');
        const searchResults = await this.knowledgeBase.searchRelevantDocuments(message);
        response = await this.generateResponseWithRAG(message, searchResults, memoryVariables.history);
        sources = searchResults.map(r => r.document.title);
        usedRAG = true;
      } else {
        // 常规对话
        this.logger.log('常规对话处理');
        response = await this.generateGeneralResponse(message, memoryVariables.history);
        usedMemory = true;
      }

      // 5. 记录AI回复到记忆
      await this.conversationMemory.addMessage(sessionId, 'assistant', response);

      const executionTime = Date.now() - startTime;

      const chatResponse: ChatResponse = {
        response,
        sessionId,
        timestamp: new Date(),
        sources,
        sqlResults,
        metadata: {
          usedMemory,
          usedRAG,
          usedSQL,
          executionTime
        }
      };

      this.logger.log(`响应生成完成，耗时: ${executionTime}ms`);
      return chatResponse;

    } catch (error) {
      this.logger.error('处理聊天消息失败:', error);
      throw error;
    }
  }

  /**
   * 分析用户意图
   */
  private async analyzeUserIntent(message: string): Promise<{
    intent: string;
    needsSQL: boolean;
    needsKnowledge: boolean;
    confidence: number;
  }> {
    const prompt = PromptTemplate.fromTemplate(`
分析用户消息的意图，判断需要使用哪些工具来回答。

用户消息: {message}

判断标准：
1. 如果用户询问具体数据、统计信息、订单查询、用户信息等，设置 needsSQL 为 true
2. 如果用户询问政策、流程、产品信息、技术支持等，设置 needsKnowledge 为 true
3. 如果是普通聊天、问候、感谢等，两者都设置为 false

请以JSON格式返回:
{{
  "intent": "意图描述",
  "needsSQL": true/false,
  "needsKnowledge": true/false,
  "confidence": 0.0-1.0
}}
    `);

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({ message });
    
    try {
      const parsed = JSON.parse(result.text);
      return {
        intent: parsed.intent || 'general',
        needsSQL: parsed.needsSQL || false,
        needsKnowledge: parsed.needsKnowledge || false,
        confidence: parsed.confidence || 0.5
      };
    } catch (error) {
      this.logger.warn('意图分析解析失败，使用默认值');
      return {
        intent: 'general',
        needsSQL: false,
        needsKnowledge: false,
        confidence: 0.5
      };
    }
  }

  /**
   * 使用SQL结果生成回复
   */
  private async generateResponseWithSQL(
    message: string,
    sqlResults: SQLQueryResult,
    conversationHistory: string
  ): Promise<string> {
    const prompt = PromptTemplate.fromTemplate(`
你是一个专业的客户服务助手。根据用户的问题和数据库查询结果，生成有帮助的回复。

用户问题: {message}

SQL查询: {sqlQuery}
查询结果: {sqlResults}
查询解释: {explanation}

对话历史: {history}

请基于查询结果回答用户的问题，要求：
1. 用友好、专业的语调
2. 准确传达查询结果的信息
3. 如果结果为空，礼貌地说明情况
4. 提供进一步的帮助建议

回复:
    `);

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      message,
      sqlQuery: sqlResults.query,
      sqlResults: JSON.stringify(sqlResults.results, null, 2),
      explanation: sqlResults.explanation,
      history: conversationHistory || ''
    });

    return result.text.trim();
  }

  /**
   * 使用RAG结果生成回复
   */
  private async generateResponseWithRAG(
    message: string,
    searchResults: SearchResult[],
    conversationHistory: string
  ): Promise<string> {
    const prompt = PromptTemplate.fromTemplate(`
你是一个专业的客户服务助手。根据用户的问题和知识库搜索结果，生成有帮助的回复。

用户问题: {message}

相关知识库内容:
{knowledgeContent}

对话历史: {history}

请基于知识库内容回答用户的问题，要求：
1. 用友好、专业的语调
2. 准确传达知识库中的信息
3. 如果知识库中没有相关信息，诚实地说明
4. 提供具体的帮助和建议

回复:
    `);

    const knowledgeContent = searchResults
      .map(result => `来源: ${result.document.title}\n内容: ${result.relevantChunk}`)
      .join('\n\n');

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      message,
      knowledgeContent,
      history: conversationHistory || ''
    });

    return result.text.trim();
  }

  /**
   * 生成常规回复
   */
  private async generateGeneralResponse(
    message: string,
    conversationHistory: string
  ): Promise<string> {
    const prompt = PromptTemplate.fromTemplate(`
你是一个友好、专业的客户服务助手。根据用户的消息和对话历史，生成合适的回复。

用户消息: {message}

对话历史: {history}

回复要求：
1. 保持友好、专业的语调
2. 根据上下文提供有意义的回复
3. 如果用户需要具体帮助，引导他们提供更多信息
4. 表现出关心和乐于助人的态度

回复:
    `);

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      message,
      history: conversationHistory || ''
    });

    return result.text.trim();
  }

  /**
   * 创建Agent工具
   */
  private async createTools(): Promise<Tool[]> {
    const tools: Tool[] = [
      // 知识库搜索工具
      {
        name: 'knowledge_search',
        description: '搜索知识库获取相关信息，用于回答关于政策、流程、产品等问题',
        func: async (query: string) => {
          const results = await this.knowledgeBase.searchRelevantDocuments(query);
          return JSON.stringify(results.map(r => ({
            title: r.document.title,
            content: r.relevantChunk,
            score: r.score
          })));
        }
      },

      // SQL查询工具
      {
        name: 'database_query',
        description: '查询数据库获取用户信息、订单状态、产品库存等数据',
        func: async (query: string) => {
          const results = await this.text2sql.naturalLanguageToSQL(query);
          return JSON.stringify(results);
        }
      },

      // 会话记忆工具
      {
        name: 'conversation_memory',
        description: '获取当前会话的历史记录和上下文',
        func: async (sessionId: string) => {
          const history = await this.conversationMemory.getConversationHistory(sessionId, 10);
          return JSON.stringify(history);
        }
      }
    ];

    return tools;
  }

  /**
   * 创建Agent提示词
   */
  private createAgentPrompt(): PromptTemplate {
    return PromptTemplate.fromTemplate(`
你是一个专业的AI客户服务助手。你的任务是帮助用户解决问题，提供准确的信息和优质的服务。

你有以下工具可以使用：
{tools}

使用以下格式：

Question: 用户的问题
Thought: 我需要思考如何回答这个问题
Action: 要使用的工具名称
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (可以重复 Thought/Action/Action Input/Observation)
Thought: 我现在知道最终答案了
Final Answer: 给用户的最终回复

重要提醒：
1. 始终保持友好、专业的态度
2. 如果不确定答案，诚实地说明
3. 尽量提供具体、有帮助的信息
4. 必要时引导用户提供更多信息

开始！

Question: {input}
Thought: {agent_scratchpad}
    `);
  }

  /**
   * 获取会话分析
   */
  async getSessionAnalysis(sessionId: string): Promise<any> {
    return await this.conversationMemory.analyzeConversation(sessionId);
  }

  /**
   * 清除会话记忆
   */
  async clearSession(sessionId: string): Promise<void> {
    await this.conversationMemory.clearMemory(sessionId);
  }

  /**
   * 获取活跃会话列表
   */
  getActiveSessions(): string[] {
    return this.conversationMemory.getActiveSessions();
  }

  /**
   * 搜索历史对话
   */
  async searchConversations(sessionId: string, query: string): Promise<any[]> {
    return await this.conversationMemory.searchConversations(sessionId, query);
  }
}