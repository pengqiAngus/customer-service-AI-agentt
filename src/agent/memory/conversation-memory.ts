import { Injectable, Logger } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import { ConversationSummaryBufferMemory } from "langchain/memory";
import { ChatOpenAI } from "@langchain/openai";
import * as fs from "fs";
import * as path from "path";

export interface ConversationMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

@Injectable()
export class ConversationMemory {
  private readonly logger = new Logger(ConversationMemory.name);
  private memoryStore = new Map<string, ConversationSummaryBufferMemory>();
  private conversationHistory = new Map<string, ConversationMessage[]>();
  private llm: ChatOpenAI;

  constructor(private configService: ConfigService) {
    this.llm = new ChatOpenAI({
      openAIApiKey: this.configService.get<string>("OPENAI_API_KEY"),
      modelName:
        this.configService.get<string>("OPENAI_MODEL") || "gpt-3.5-turbo",
      temperature: 0.1,
    });
  }

  /**
   * 获取或创建会话的记忆实例
   */
  async getMemory(sessionId: string): Promise<ConversationSummaryBufferMemory> {
    if (!this.memoryStore.has(sessionId)) {
      const memory = new ConversationSummaryBufferMemory({
        llm: this.llm,
        maxTokenLimit: this.configService.get<number>("MAX_MEMORY_SIZE") || 100,
        returnMessages: true,
      });

      // 加载历史对话
      await this.loadConversationHistory(sessionId, memory);
      this.memoryStore.set(sessionId, memory);

      this.logger.log(`为会话 ${sessionId} 创建新的记忆实例`);
    }

    return this.memoryStore.get(sessionId);
  }

  /**
   * 添加消息到记忆中
   */
  async addMessage(
    sessionId: string,
    role: "user" | "assistant",
    content: string,
    metadata?: Record<string, any>
  ): Promise<void> {
    const memory = await this.getMemory(sessionId);

    const message: ConversationMessage = {
      id: this.generateMessageId(),
      sessionId,
      role,
      content,
      timestamp: new Date(),
      metadata,
    };

    // 添加到历史记录
    if (!this.conversationHistory.has(sessionId)) {
      this.conversationHistory.set(sessionId, []);
    }
    this.conversationHistory.get(sessionId).push(message);

    // 添加到langchain记忆
    if (role === "user") {
      await memory.saveContext({ input: content }, { output: "" });
    } else {
      // 获取最后一个用户消息
      const history = this.conversationHistory.get(sessionId);
      const lastUserMessage = history
        .slice()
        .reverse()
        .find((m) => m.role === "user");

      if (lastUserMessage) {
        await memory.saveContext(
          { input: lastUserMessage.content },
          { output: content }
        );
      }
    }

    // 持久化保存
    await this.saveConversationHistory(sessionId);

    this.logger.log(`已添加 ${role} 消息到会话 ${sessionId}`);
  }

  /**
   * 获取记忆中的上下文
   */
  async getMemoryVariables(sessionId: string): Promise<Record<string, any>> {
    const memory = await this.getMemory(sessionId);
    return await memory.loadMemoryVariables({});
  }

  /**
   * 获取会话历史
   */
  async getConversationHistory(
    sessionId: string,
    limit?: number
  ): Promise<ConversationMessage[]> {
    const history = this.conversationHistory.get(sessionId) || [];

    if (limit) {
      return history.slice(-limit);
    }

    return history;
  }

  /**
   * 清除会话记忆
   */
  async clearMemory(sessionId: string): Promise<void> {
    this.memoryStore.delete(sessionId);
    this.conversationHistory.delete(sessionId);

    // 删除持久化文件
    const historyPath = this.getHistoryFilePath(sessionId);
    if (fs.existsSync(historyPath)) {
      fs.unlinkSync(historyPath);
    }

    this.logger.log(`已清除会话 ${sessionId} 的记忆`);
  }

  /**
   * 获取所有活跃会话
   */
  getActiveSessions(): string[] {
    return Array.from(this.memoryStore.keys());
  }

  /**
   * 搜索历史对话
   */
  async searchConversations(
    sessionId: string,
    query: string,
    limit: number = 5
  ): Promise<ConversationMessage[]> {
    const history = this.conversationHistory.get(sessionId) || [];

    // 简单的文本搜索（实际应用中可以使用更sophisticated的搜索方法）
    const results = history.filter((message) =>
      message.content.toLowerCase().includes(query.toLowerCase())
    );

    return results.slice(-limit);
  }

  /**
   * 获取会话摘要
   */
  async getConversationSummary(sessionId: string): Promise<string> {
    const memory = await this.getMemory(sessionId);
    const variables = await memory.loadMemoryVariables({});

    return variables.history || "";
  }

  /**
   * 分析对话情感和意图
   */
  async analyzeConversation(sessionId: string): Promise<{
    sentiment: string;
    intent: string;
    topics: string[];
    totalMessages: number;
  }> {
    const history = this.conversationHistory.get(sessionId) || [];

    // 简单的分析（实际应用中可以使用更sophisticated的NLP技术）
    const userMessages = history.filter((m) => m.role === "user");
    const topics = this.extractTopics(userMessages.map((m) => m.content));

    return {
      sentiment: this.analyzeSentiment(userMessages.map((m) => m.content)),
      intent: this.extractIntent(userMessages.map((m) => m.content)),
      topics,
      totalMessages: history.length,
    };
  }

  /**
   * 加载历史对话
   */
  private async loadConversationHistory(
    sessionId: string,
    memory: ConversationSummaryBufferMemory
  ): Promise<void> {
    const historyPath = this.getHistoryFilePath(sessionId);

    if (fs.existsSync(historyPath)) {
      try {
        const historyData = fs.readFileSync(historyPath, "utf8");
        const history: ConversationMessage[] = JSON.parse(historyData);

        this.conversationHistory.set(sessionId, history);

        // 重建langchain记忆
        for (let i = 0; i < history.length; i += 2) {
          const userMsg = history[i];
          const assistantMsg = history[i + 1];

          if (
            userMsg &&
            assistantMsg &&
            userMsg.role === "user" &&
            assistantMsg.role === "assistant"
          ) {
            await memory.saveContext(
              { input: userMsg.content },
              { output: assistantMsg.content }
            );
          }
        }

        this.logger.log(
          `为会话 ${sessionId} 加载了 ${history.length} 条历史消息`
        );
      } catch (error) {
        this.logger.error(`加载会话历史失败: ${error.message}`);
      }
    }
  }

  /**
   * 保存对话历史
   */
  private async saveConversationHistory(sessionId: string): Promise<void> {
    const history = this.conversationHistory.get(sessionId);
    if (!history) return;

    const historyPath = this.getHistoryFilePath(sessionId);
    const historyDir = path.dirname(historyPath);

    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }

    try {
      fs.writeFileSync(historyPath, JSON.stringify(history, null, 2));
    } catch (error) {
      this.logger.error(`保存会话历史失败: ${error.message}`);
    }
  }

  /**
   * 获取历史文件路径
   */
  private getHistoryFilePath(sessionId: string): string {
    return path.join("./data/conversations", `${sessionId}.json`);
  }

  /**
   * 生成消息ID
   */
  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 提取话题
   */
  private extractTopics(messages: string[]): string[] {
    const allText = messages.join(" ").toLowerCase();

    // 简单的关键词提取
    const keywords = [
      "产品",
      "订单",
      "退款",
      "保修",
      "技术支持",
      "价格",
      "配送",
      "客服",
    ];
    return keywords.filter((keyword) => allText.includes(keyword));
  }

  /**
   * 分析情感
   */
  private analyzeSentiment(messages: string[]): string {
    const allText = messages.join(" ").toLowerCase();

    const positiveWords = ["好", "满意", "喜欢", "不错", "棒"];
    const negativeWords = ["差", "不满", "问题", "故障", "糟糕"];

    const positiveCount = positiveWords.filter((word) =>
      allText.includes(word)
    ).length;
    const negativeCount = negativeWords.filter((word) =>
      allText.includes(word)
    ).length;

    if (positiveCount > negativeCount) return "positive";
    if (negativeCount > positiveCount) return "negative";
    return "neutral";
  }

  /**
   * 提取意图
   */
  private extractIntent(messages: string[]): string {
    const allText = messages.join(" ").toLowerCase();

    if (allText.includes("退款") || allText.includes("退货")) return "refund";
    if (allText.includes("订单") || allText.includes("购买")) return "order";
    if (allText.includes("技术") || allText.includes("故障"))
      return "technical_support";
    if (allText.includes("价格") || allText.includes("优惠")) return "pricing";

    return "general_inquiry";
  }
}
