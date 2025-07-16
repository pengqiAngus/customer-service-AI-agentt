import {
  Injectable,
  Logger,
  Optional,
  Inject,
  forwardRef,
} from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import { ConversationMemory } from "./memory/conversation-memory";
import { KnowledgeBase, SearchResult } from "./rag/knowledge-base";
import { Text2SQL, SQLQueryResult } from "./sql/text2sql";
import { AgentWorkflow, WorkflowState } from "./workflow/agent-workflow";

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
    workflowPath: string[];
    intent: string;
    confidence: number;
  };
}

@Injectable()
export class AIAgentService {
  private readonly logger = new Logger(AIAgentService.name);

  constructor(
    private configService: ConfigService,
    private conversationMemory: ConversationMemory,
    private knowledgeBase: KnowledgeBase,
    @Optional() @Inject(forwardRef(() => Text2SQL)) private text2sql?: Text2SQL,
    private agentWorkflow?: AgentWorkflow
  ) {}

  /**
   * 处理聊天消息 - 使用LangGraph工作流
   */
  async chat(chatMessage: ChatMessage): Promise<ChatResponse> {
    const { message, sessionId, userId } = chatMessage;

    this.logger.log(`处理用户消息: ${message} (会话: ${sessionId})`);

    try {
      // 1. 记录用户消息到记忆
      await this.conversationMemory.addMessage(sessionId, "user", message, {
        userId,
      });

      // 2. 执行LangGraph工作流
      const workflowResult = await this.agentWorkflow.executeWorkflow({
        message,
        sessionId,
        userId,
      });

      // 3. 记录AI回复到记忆
      await this.conversationMemory.addMessage(
        sessionId,
        "assistant",
        workflowResult.response
      );

      // 4. 构建响应
      const chatResponse: ChatResponse = {
        response: workflowResult.response,
        sessionId,
        timestamp: new Date(),
        sources: workflowResult.sources,
        sqlResults: workflowResult.sqlResults,
        metadata: {
          usedMemory: workflowResult.metadata.usedMemory,
          usedRAG: workflowResult.metadata.usedRAG,
          usedSQL: workflowResult.metadata.usedSQL,
          executionTime: workflowResult.executionTime,
          workflowPath: workflowResult.metadata.workflowPath,
          intent: workflowResult.intent,
          confidence: workflowResult.confidence,
        },
      };

      this.logger.log(
        `✅ 工作流执行完成，路径: ${workflowResult.metadata.workflowPath.join(" -> ")}`
      );
      return chatResponse;
    } catch (error) {
      this.logger.error("处理聊天消息失败:", error);
      throw error;
    }
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

  /**
   * 获取工作流统计信息
   */
  getWorkflowStats(): any {
    return this.agentWorkflow.getWorkflowStats();
  }
}
