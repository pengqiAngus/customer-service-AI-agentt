import { 
  Controller, 
  Post, 
  Get, 
  Delete,
  Body, 
  Param, 
  Query,
  Logger 
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBody } from '@nestjs/swagger';
import { AIAgentService, ChatMessage, ChatResponse } from './ai-agent.service';
import { KnowledgeBase, KnowledgeDocument } from './rag/knowledge-base';
import { Text2SQL, SQLQueryResult } from './sql/text2sql';

export class ChatRequestDto {
  message: string;
  sessionId: string;
  userId?: string;
}

export class AddDocumentDto {
  title: string;
  content: string;
  category: string;
  url?: string;
  metadata?: Record<string, any>;
}

@ApiTags('AI客户助手')
@Controller('agent')
export class AgentController {
  private readonly logger = new Logger(AgentController.name);

  constructor(
    private readonly aiAgentService: AIAgentService,
    private readonly knowledgeBase: KnowledgeBase,
    private readonly text2sql: Text2SQL,
  ) {}

  @Post('chat')
  @ApiOperation({ summary: '与AI客户助手对话' })
  @ApiResponse({ status: 200, description: '对话成功' })
  @ApiBody({ type: ChatRequestDto })
  async chat(@Body() chatRequest: ChatRequestDto): Promise<ChatResponse> {
    this.logger.log(`收到聊天请求: ${chatRequest.message}`);
    
    const chatMessage: ChatMessage = {
      message: chatRequest.message,
      sessionId: chatRequest.sessionId,
      userId: chatRequest.userId,
    };

    return await this.aiAgentService.chat(chatMessage);
  }

  @Get('session/:sessionId/analysis')
  @ApiOperation({ summary: '获取会话分析' })
  @ApiResponse({ status: 200, description: '获取分析成功' })
  async getSessionAnalysis(@Param('sessionId') sessionId: string): Promise<any> {
    return await this.aiAgentService.getSessionAnalysis(sessionId);
  }

  @Delete('session/:sessionId')
  @ApiOperation({ summary: '清除会话记忆' })
  @ApiResponse({ status: 200, description: '清除成功' })
  async clearSession(@Param('sessionId') sessionId: string): Promise<{ message: string }> {
    await this.aiAgentService.clearSession(sessionId);
    return { message: '会话记忆已清除' };
  }

  @Get('sessions/active')
  @ApiOperation({ summary: '获取活跃会话列表' })
  @ApiResponse({ status: 200, description: '获取成功' })
  async getActiveSessions(): Promise<{ sessions: string[] }> {
    const sessions = this.aiAgentService.getActiveSessions();
    return { sessions };
  }

  @Get('session/:sessionId/search')
  @ApiOperation({ summary: '搜索历史对话' })
  @ApiResponse({ status: 200, description: '搜索成功' })
  async searchConversations(
    @Param('sessionId') sessionId: string,
    @Query('query') query: string
  ): Promise<any> {
    return await this.aiAgentService.searchConversations(sessionId, query);
  }

  // 知识库管理接口
  @Post('knowledge/documents')
  @ApiOperation({ summary: '添加知识库文档' })
  @ApiResponse({ status: 200, description: '添加成功' })
  async addDocument(@Body() documentDto: AddDocumentDto): Promise<{ documentId: string }> {
    const documentId = await this.knowledgeBase.addDocument({
      title: documentDto.title,
      content: documentDto.content,
      category: documentDto.category,
      url: documentDto.url,
      metadata: documentDto.metadata || {}
    });

    return { documentId };
  }

  @Get('knowledge/documents')
  @ApiOperation({ summary: '获取所有知识库文档' })
  @ApiResponse({ status: 200, description: '获取成功' })
  async getAllDocuments(): Promise<{ documents: KnowledgeDocument[] }> {
    const documents = await this.knowledgeBase.getAllDocuments();
    return { documents };
  }

  @Get('knowledge/documents/category/:category')
  @ApiOperation({ summary: '按类别获取知识库文档' })
  @ApiResponse({ status: 200, description: '获取成功' })
  async getDocumentsByCategory(
    @Param('category') category: string
  ): Promise<{ documents: KnowledgeDocument[] }> {
    const documents = await this.knowledgeBase.getDocumentsByCategory(category);
    return { documents };
  }

  @Post('knowledge/search')
  @ApiOperation({ summary: '搜索知识库' })
  @ApiResponse({ status: 200, description: '搜索成功' })
  async searchKnowledge(
    @Body() searchRequest: { query: string; limit?: number }
  ): Promise<any> {
    const results = await this.knowledgeBase.searchRelevantDocuments(
      searchRequest.query,
      searchRequest.limit || 5
    );
    return { results };
  }

  @Delete('knowledge/documents/:documentId')
  @ApiOperation({ summary: '删除知识库文档' })
  @ApiResponse({ status: 200, description: '删除成功' })
  async deleteDocument(@Param('documentId') documentId: string): Promise<{ message: string }> {
    const success = await this.knowledgeBase.deleteDocument(documentId);
    
    if (success) {
      return { message: '文档删除成功' };
    } else {
      return { message: '文档不存在' };
    }
  }

  // SQL查询接口
  @Post('sql/query')
  @ApiOperation({ summary: '自然语言转SQL查询' })
  @ApiResponse({ status: 200, description: '查询成功' })
  async naturalLanguageQuery(
    @Body() queryRequest: { question: string }
  ): Promise<SQLQueryResult> {
    return await this.text2sql.naturalLanguageToSQL(queryRequest.question);
  }

  @Get('sql/schemas')
  @ApiOperation({ summary: '获取数据库表结构' })
  @ApiResponse({ status: 200, description: '获取成功' })
  async getTableSchemas(): Promise<any> {
    const schemas = await this.text2sql.getTableSchemas();
    return { schemas };
  }

  @Post('sql/direct')
  @ApiOperation({ summary: '直接执行SQL查询（仅用于测试）' })
  @ApiResponse({ status: 200, description: '执行成功' })
  async executeDirectSQL(
    @Body() sqlRequest: { query: string }
  ): Promise<{ results: any[] }> {
    const results = await this.text2sql.executeDirectSQL(sqlRequest.query);
    return { results };
  }

  // 健康检查和状态接口
  @Get('health')
  @ApiOperation({ summary: '健康检查' })
  @ApiResponse({ status: 200, description: '服务正常' })
  async healthCheck(): Promise<{ 
    status: string; 
    timestamp: string;
    services: Record<string, boolean>; 
  }> {
    // 简单的健康检查
    const services = {
      aiAgent: true,
      knowledgeBase: true,
      text2sql: true,
      memory: true
    };

    return {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services
    };
  }

  @Get('stats')
  @ApiOperation({ summary: '获取服务统计信息' })
  @ApiResponse({ status: 200, description: '获取成功' })
  async getStats(): Promise<any> {
    const activeSessions = this.aiAgentService.getActiveSessions();
    const documents = await this.knowledgeBase.getAllDocuments();
    const schemas = await this.text2sql.getTableSchemas();

    return {
      activeSessions: activeSessions.length,
      totalDocuments: documents.length,
      documentsByCategory: this.groupDocumentsByCategory(documents),
      databaseTables: schemas.length,
      timestamp: new Date().toISOString()
    };
  }

  private groupDocumentsByCategory(documents: KnowledgeDocument[]): Record<string, number> {
    const grouped: Record<string, number> = {};
    
    documents.forEach(doc => {
      grouped[doc.category] = (grouped[doc.category] || 0) + 1;
    });

    return grouped;
  }
}