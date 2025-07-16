import {
  Injectable,
  Logger,
  Optional,
  Inject,
  forwardRef,
} from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, Annotation, END, START } from "@langchain/langgraph";
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { ConversationMemory } from "../memory/conversation-memory";
import { KnowledgeBase, SearchResult } from "../rag/knowledge-base";
import { Text2SQL, SQLQueryResult } from "../sql/text2sql";

export interface WorkflowState {
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

@Injectable()
export class AgentWorkflow {
  private readonly logger = new Logger(AgentWorkflow.name);
  private llm: ChatOpenAI;
  private workflow: any;

  constructor(
    private configService: ConfigService,
    private conversationMemory: ConversationMemory,
    private knowledgeBase: KnowledgeBase,
    @Optional() @Inject(forwardRef(() => Text2SQL)) private text2sql?: Text2SQL
  ) {
    this.initializeWorkflow();
  }

  /**
   * 初始化LangGraph工作流
   */
  private async initializeWorkflow(): Promise<void> {
    this.logger.log("初始化LangGraph工作流...");

    this.llm = new ChatOpenAI({
      openAIApiKey: this.configService.get<string>("OPENAI_API_KEY"),
      modelName:
        this.configService.get<string>("OPENAI_MODEL") || "gpt-3.5-turbo",
      temperature: 0.1,
    });

    // 1. 定义 Annotation
    const WorkflowStateAnnotation = Annotation.Root({
      message: Annotation<string>(),
      sessionId: Annotation<string>(),
      userId: Annotation<string | undefined>({
        value: (_old, update) => update,
        default: undefined,
      }),
      intent: Annotation<string>(),
      needsSQL: Annotation<boolean>(),
      needsKnowledge: Annotation<boolean>(),
      confidence: Annotation<number>(),
      conversationHistory: Annotation<string>(),
      response: Annotation<string | undefined>({
        value: (_old, update) => update,
        default: undefined,
      }),
      sources: Annotation<string[]>({
        value: (_old, update) => update,
        default: () => [],
      }),
      sqlResults: Annotation<any | undefined>({
        value: (_old, update) => update,
        default: undefined,
      }),
      searchResults: Annotation<any[]>({
        value: (_old, update) => update,
        default: () => [],
      }),
      executionTime: Annotation<number>(),
      metadata: Annotation<{
        usedMemory: boolean;
        usedRAG: boolean;
        usedSQL: boolean;
        workflowPath: string[];
      }>({
        value: (_old, update) => update,
        default: () => ({
          usedMemory: false,
          usedRAG: false,
          usedSQL: false,
          workflowPath: [],
        }),
      }),
    });

    // 2. 初始化 StateGraph
    this.workflow = new StateGraph(WorkflowStateAnnotation);

    // 3. 添加节点
    this.workflow.addNode("intent_analysis", this.intentAnalysis.bind(this));
    this.workflow.addNode("sql_processing", this.sqlProcessing.bind(this));
    this.workflow.addNode(
      "knowledge_processing",
      this.knowledgeProcessing.bind(this)
    );
    this.workflow.addNode(
      "general_conversation",
      this.generalConversation.bind(this)
    );
    this.workflow.addNode(
      "response_generation",
      this.responseGeneration.bind(this)
    );

    // 4. 设置入口点（用addEdge连接__start__）
    this.workflow.addEdge("__start__", "intent_analysis");

    // 5. 添加条件边和普通边（保持原有逻辑）
    this.workflow.addConditionalEdges(
      "intent_analysis",
      this.routeBasedOnIntent.bind(this),
      {
        sql: "sql_processing",
        knowledge: "knowledge_processing",
        general: "general_conversation",
      }
    );
    this.workflow.addEdge("sql_processing", "response_generation");
    this.workflow.addEdge("knowledge_processing", "response_generation");
    this.workflow.addEdge("general_conversation", "response_generation");
    this.workflow.addEdge("response_generation", "__end__");

    // 6. 编译
    this.workflow = this.workflow.compile() as any;

    this.logger.log("LangGraph工作流初始化完成");
  }

  /**
   * 意图分析节点
   */
  private async intentAnalysis(
    state: WorkflowState
  ): Promise<Partial<WorkflowState>> {
    this.logger.log("🔍 开始意图分析...");

    const prompt = PromptTemplate.fromTemplate(`
分析用户消息的意图，判断需要使用哪种处理方式。

用户消息: {message}
对话历史: {history}

分析标准：
1. SQL查询：用户询问具体数据、统计信息、订单查询、用户信息等
2. 知识库查询：用户询问政策、流程、产品信息、技术支持等
3. 常规对话：普通聊天、问候、感谢等

请以JSON格式返回:
{{
  "intent": "意图描述",
  "needsSQL": true/false,
  "needsKnowledge": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "分析理由"
}}
    `) as any;

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      message: state.message,
      history: state.conversationHistory || "",
    });

    try {
      const analysis = JSON.parse(result.text);

      const updatedState: Partial<WorkflowState> = {
        intent: analysis.intent || "general",
        needsSQL: analysis.needsSQL || false,
        needsKnowledge: analysis.needsKnowledge || false,
        confidence: analysis.confidence || 0.5,
        metadata: {
          ...state.metadata,
          workflowPath: [...state.metadata.workflowPath, "intent_analysis"],
        },
      };

      this.logger.log(
        `意图分析结果: ${analysis.intent} (置信度: ${analysis.confidence})`
      );
      return updatedState;
    } catch (error) {
      this.logger.warn("意图分析解析失败，使用默认值");
      return {
        intent: "general",
        needsSQL: false,
        needsKnowledge: false,
        confidence: 0.5,
        metadata: {
          ...state.metadata,
          workflowPath: [...state.metadata.workflowPath, "intent_analysis"],
        },
      };
    }
  }

  /**
   * SQL处理节点
   */
  private async sqlProcessing(
    state: WorkflowState
  ): Promise<Partial<WorkflowState>> {
    this.logger.log("🗄️ 开始SQL处理...");

    try {
      const sqlResults = await this.text2sql.naturalLanguageToSQL(
        state.message
      );

      return {
        sqlResults,
        metadata: {
          ...state.metadata,
          usedSQL: true,
          workflowPath: [...state.metadata.workflowPath, "sql_processing"],
        },
      };
    } catch (error) {
      this.logger.error("SQL处理失败:", error);
      throw error;
    }
  }

  /**
   * 知识库处理节点
   */
  private async knowledgeProcessing(
    state: WorkflowState
  ): Promise<Partial<WorkflowState>> {
    this.logger.log("📚 开始知识库处理...");

    try {
      const searchResults = await this.knowledgeBase.searchRelevantDocuments(
        state.message
      );
      const sources = searchResults.map((r) => r.document.title);

      return {
        searchResults,
        sources,
        metadata: {
          ...state.metadata,
          usedRAG: true,
          workflowPath: [
            ...state.metadata.workflowPath,
            "knowledge_processing",
          ],
        },
      };
    } catch (error) {
      this.logger.error("知识库处理失败:", error);
      throw error;
    }
  }

  /**
   * 常规对话处理节点
   */
  private async generalConversation(
    state: WorkflowState
  ): Promise<Partial<WorkflowState>> {
    this.logger.log("💬 开始常规对话处理...");

    return {
      metadata: {
        ...state.metadata,
        usedMemory: true,
        workflowPath: [...state.metadata.workflowPath, "general_conversation"],
      },
    };
  }

  /**
   * 响应生成节点
   */
  private async responseGeneration(
    state: WorkflowState
  ): Promise<Partial<WorkflowState>> {
    this.logger.log("🤖 生成最终响应...");

    let response: string;

    if (state.sqlResults) {
      // 基于SQL结果生成响应
      response = await this.generateResponseWithSQL(
        state.message,
        state.sqlResults,
        state.conversationHistory
      );
    } else if (state.searchResults && state.searchResults.length > 0) {
      // 基于知识库结果生成响应
      response = await this.generateResponseWithRAG(
        state.message,
        state.searchResults,
        state.conversationHistory
      );
    } else {
      // 生成常规响应
      response = await this.generateGeneralResponse(
        state.message,
        state.conversationHistory
      );
    }

    return {
      response,
      metadata: {
        ...state.metadata,
        workflowPath: [...state.metadata.workflowPath, "response_generation"],
      },
    };
  }

  /**
   * 基于意图的路由决策
   */
  private routeBasedOnIntent(state: WorkflowState): string {
    this.logger.log(
      `🛤️ 路由决策: ${state.intent} (SQL: ${state.needsSQL}, Knowledge: ${state.needsKnowledge})`
    );

    if (state.needsSQL) {
      return "sql";
    } else if (state.needsKnowledge) {
      return "knowledge";
    } else {
      return "general";
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
    `) as any;

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      message,
      sqlQuery: sqlResults.query,
      sqlResults: JSON.stringify(sqlResults.results, null, 2),
      explanation: sqlResults.explanation,
      history: conversationHistory || "",
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
    `) as any;

    const knowledgeContent = searchResults
      .map(
        (result) =>
          `来源: ${result.document.title}\n内容: ${result.relevantChunk}`
      )
      .join("\n\n");

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      message,
      knowledgeContent,
      history: conversationHistory || "",
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
    `) as any;

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      message,
      history: conversationHistory || "",
    });

    return result.text.trim();
  }

  /**
   * 执行工作流
   */
  async executeWorkflow(
    initialState: Partial<WorkflowState>
  ): Promise<WorkflowState> {
    const startTime = Date.now();

    this.logger.log(`🚀 开始执行工作流: ${initialState.message}`);

    try {
      // 获取对话历史
      const memoryVariables = await this.conversationMemory.getMemoryVariables(
        initialState.sessionId
      );

      // 构建初始状态
      const state: WorkflowState = {
        message: initialState.message,
        sessionId: initialState.sessionId,
        userId: initialState.userId,
        intent: "",
        needsSQL: false,
        needsKnowledge: false,
        confidence: 0,
        conversationHistory: memoryVariables.history || "",
        response: "",
        sources: [],
        sqlResults: undefined,
        searchResults: [],
        executionTime: 0,
        metadata: {
          usedMemory: false,
          usedRAG: false,
          usedSQL: false,
          workflowPath: [],
        },
      };

      // 执行工作流
      const result = await this.workflow.invoke(state);

      const executionTime = Date.now() - startTime;

      this.logger.log(`✅ 工作流执行完成，耗时: ${executionTime}ms`);
      this.logger.log(
        `🛤️ 执行路径: ${result.metadata.workflowPath.join(" -> ")}`
      );

      return {
        ...result,
        executionTime,
      };
    } catch (error) {
      this.logger.error("工作流执行失败:", error);
      throw error;
    }
  }

  /**
   * 获取工作流统计信息
   */
  getWorkflowStats(): any {
    return {
      nodes: [
        "intent_analysis",
        "sql_processing",
        "knowledge_processing",
        "general_conversation",
        "response_generation",
      ],
      edges: [
        "intent_analysis -> sql_processing",
        "intent_analysis -> knowledge_processing",
        "intent_analysis -> general_conversation",
        "sql_processing -> response_generation",
        "knowledge_processing -> response_generation",
        "general_conversation -> response_generation",
        "response_generation -> END",
      ],
      conditionalEdges: ["intent_analysis (基于意图路由)"],
      description: "智能意图识别和响应生成工作流",
    };
  }
}
