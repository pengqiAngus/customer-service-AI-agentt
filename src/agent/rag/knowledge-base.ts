import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { OpenAIEmbeddings } from '@langchain/openai';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from 'langchain/document';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

export interface KnowledgeDocument {
  id: string;
  title: string;
  content: string;
  url?: string;
  category: string;
  metadata: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
}

export interface SearchResult {
  document: KnowledgeDocument;
  score: number;
  relevantChunk: string;
}

@Injectable()
export class KnowledgeBase {
  private readonly logger = new Logger(KnowledgeBase.name);
  private vectorStore: FaissStore;
  private embeddings: OpenAIEmbeddings;
  private textSplitter: RecursiveCharacterTextSplitter;
  private documents: Map<string, KnowledgeDocument> = new Map();

  constructor(private configService: ConfigService) {
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: this.configService.get<string>('OPENAI_API_KEY'),
      modelName: this.configService.get<string>('EMBEDDING_MODEL') || 'text-embedding-ada-002',
    });

    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    this.initializeKnowledgeBase();
  }

  /**
   * 初始化知识库
   */
  private async initializeKnowledgeBase(): Promise<void> {
    this.logger.log('初始化知识库...');
    
    try {
      // 尝试加载已存在的向量存储
      await this.loadVectorStore();
      
      // 如果没有现有的向量存储，创建新的
      if (!this.vectorStore) {
        await this.createVectorStore();
      }
      
      // 加载知识文档
      await this.loadKnowledgeDocuments();
      
      this.logger.log('知识库初始化完成');
    } catch (error) {
      this.logger.error('知识库初始化失败:', error);
      throw error;
    }
  }

  /**
   * 添加文档到知识库
   */
  async addDocument(document: Omit<KnowledgeDocument, 'id' | 'createdAt' | 'updatedAt'>): Promise<string> {
    const doc: KnowledgeDocument = {
      ...document,
      id: this.generateDocumentId(),
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    // 分割文档内容
    const textChunks = await this.textSplitter.splitText(doc.content);
    
    // 创建Document对象
    const documents = textChunks.map(chunk => new Document({
      pageContent: chunk,
      metadata: {
        documentId: doc.id,
        title: doc.title,
        category: doc.category,
        url: doc.url,
        ...doc.metadata
      }
    }));

    // 添加到向量存储
    await this.vectorStore.addDocuments(documents);
    
    // 保存文档信息
    this.documents.set(doc.id, doc);
    await this.saveKnowledgeDocuments();
    await this.saveVectorStore();

    this.logger.log(`已添加文档: ${doc.title} (${textChunks.length} 个块)`);
    return doc.id;
  }

  /**
   * 从URL批量加载文档
   */
  async loadDocumentsFromUrl(): Promise<void> {
    const knowledgeUrl = this.configService.get<string>('KNOWLEDGE_BASE_URL');
    this.logger.log(`从 ${knowledgeUrl} 加载知识文档...`);

    try {
      // 模拟从URL加载文档（实际应用中会从真实API获取）
      const mockDocuments: Omit<KnowledgeDocument, 'id' | 'createdAt' | 'updatedAt'>[] = [
        {
          title: '产品保修政策',
          content: `
            我们的产品保修政策如下：
            
            1. 保修期限：所有产品享受1年免费保修服务
            2. 保修范围：
               - 制造缺陷导致的故障
               - 正常使用条件下的质量问题
               - 非人为损坏的硬件故障
            
            3. 不在保修范围内的情况：
               - 人为损坏（摔落、液体泼洒等）
               - 自然灾害造成的损坏
               - 私自拆解或修理
               - 超过保修期的产品
            
            4. 保修流程：
               - 联系客服提供购买凭证
               - 技术人员诊断问题
               - 确认在保修范围内后安排维修或更换
               - 维修完成后进行质量检测
            
            5. 联系方式：
               - 客服热线：400-123-4567
               - 邮箱：support@company.com
               - 工作时间：周一至周五 9:00-18:00
          `,
          category: '售后服务',
          metadata: { priority: 'high', lastReviewed: '2024-01-01' }
        },
        {
          title: '订单处理流程',
          content: `
            订单处理的完整流程：
            
            1. 订单确认：
               - 核实订单信息和收货地址
               - 确认库存状态
               - 生成订单号
            
            2. 支付处理：
               - 支持多种支付方式（微信、支付宝、银行卡）
               - 支付确认后锁定库存
               - 开具发票（如需要）
            
            3. 备货和包装：
               - 仓库拣货
               - 质量检查
               - 专业包装
            
            4. 物流配送：
               - 选择最优配送路线
               - 提供物流跟踪号
               - 配送状态实时更新
            
            5. 签收确认：
               - 用户签收确认
               - 系统自动更新订单状态
               - 发送确认邮件/短信
            
            6. 订单修改和取消：
               - 未发货订单可以修改或取消
               - 已发货订单需要联系客服处理
               - 取消订单退款3-5个工作日到账
          `,
          category: '订单管理',
          metadata: { priority: 'medium', lastReviewed: '2024-01-15' }
        },
        {
          title: '技术支持指南',
          content: `
            技术支持服务内容：
            
            1. 远程诊断：
               - 通过远程软件诊断问题
               - 实时屏幕共享协助
               - 系统性能检测
            
            2. 故障排除：
               - 硬件故障诊断
               - 软件问题解决
               - 网络连接问题
               - 驱动程序更新
            
            3. 使用指导：
               - 产品使用教程
               - 功能设置说明
               - 最佳实践建议
               - 定期维护指导
            
            4. 升级和更新：
               - 系统更新通知
               - 功能升级说明
               - 兼容性检查
               - 数据备份建议
            
            5. 24/7技术热线：
               - 紧急故障快速响应
               - 专业工程师在线支持
               - 多语言技术支持
               - 问题跟踪和反馈
            
            6. 常见问题解答：
               - 设备无法启动：检查电源连接
               - 性能缓慢：清理临时文件，重启设备
               - 网络连接问题：检查网络设置和信号强度
          `,
          category: '技术支持',
          metadata: { priority: 'high', lastReviewed: '2024-01-20' }
        }
      ];

      // 批量添加文档
      for (const doc of mockDocuments) {
        await this.addDocument(doc);
      }

      this.logger.log(`成功加载 ${mockDocuments.length} 个知识文档`);
      
    } catch (error) {
      this.logger.error('加载知识文档失败:', error);
      throw error;
    }
  }

  /**
   * 搜索相关文档
   */
  async searchRelevantDocuments(
    query: string, 
    limit: number = 5,
    threshold: number = 0.7
  ): Promise<SearchResult[]> {
    if (!this.vectorStore) {
      throw new Error('向量存储未初始化');
    }

    try {
      // 执行相似性搜索
      const results = await this.vectorStore.similaritySearchWithScore(query, limit);
      
      const searchResults: SearchResult[] = [];
      
      for (const [doc, score] of results) {
        if (score >= threshold) {
          const documentId = doc.metadata.documentId;
          const document = this.documents.get(documentId);
          
          if (document) {
            searchResults.push({
              document,
              score,
              relevantChunk: doc.pageContent
            });
          }
        }
      }

      this.logger.log(`搜索 "${query}" 找到 ${searchResults.length} 个相关结果`);
      return searchResults;
      
    } catch (error) {
      this.logger.error('搜索失败:', error);
      throw error;
    }
  }

  /**
   * 获取文档内容
   */
  async getDocument(documentId: string): Promise<KnowledgeDocument | null> {
    return this.documents.get(documentId) || null;
  }

  /**
   * 更新文档
   */
  async updateDocument(
    documentId: string, 
    updates: Partial<Omit<KnowledgeDocument, 'id' | 'createdAt'>>
  ): Promise<boolean> {
    const document = this.documents.get(documentId);
    if (!document) {
      return false;
    }

    const updatedDocument = {
      ...document,
      ...updates,
      updatedAt: new Date()
    };

    this.documents.set(documentId, updatedDocument);
    await this.saveKnowledgeDocuments();

    // 如果内容更新了，需要重新建立向量索引
    if (updates.content) {
      await this.rebuildVectorIndex();
    }

    this.logger.log(`已更新文档: ${documentId}`);
    return true;
  }

  /**
   * 删除文档
   */
  async deleteDocument(documentId: string): Promise<boolean> {
    const document = this.documents.get(documentId);
    if (!document) {
      return false;
    }

    this.documents.delete(documentId);
    await this.saveKnowledgeDocuments();
    await this.rebuildVectorIndex();

    this.logger.log(`已删除文档: ${documentId}`);
    return true;
  }

  /**
   * 获取所有文档列表
   */
  async getAllDocuments(): Promise<KnowledgeDocument[]> {
    return Array.from(this.documents.values());
  }

  /**
   * 按类别获取文档
   */
  async getDocumentsByCategory(category: string): Promise<KnowledgeDocument[]> {
    return Array.from(this.documents.values()).filter(doc => doc.category === category);
  }

  /**
   * 创建新的向量存储
   */
  private async createVectorStore(): Promise<void> {
    this.logger.log('创建新的向量存储...');
    
    // 创建空的向量存储
    this.vectorStore = await FaissStore.fromTexts(
      ['初始化文档'],
      [{ type: 'init' }],
      this.embeddings
    );
  }

  /**
   * 加载向量存储
   */
  private async loadVectorStore(): Promise<void> {
    const vectorStorePath = this.configService.get<string>('VECTOR_STORE_PATH');
    
    if (fs.existsSync(vectorStorePath)) {
      try {
        this.vectorStore = await FaissStore.load(vectorStorePath, this.embeddings);
        this.logger.log('成功加载现有向量存储');
      } catch (error) {
        this.logger.warn('加载向量存储失败，将创建新的:', error.message);
      }
    }
  }

  /**
   * 保存向量存储
   */
  private async saveVectorStore(): Promise<void> {
    const vectorStorePath = this.configService.get<string>('VECTOR_STORE_PATH');
    const vectorStoreDir = path.dirname(vectorStorePath);
    
    if (!fs.existsSync(vectorStoreDir)) {
      fs.mkdirSync(vectorStoreDir, { recursive: true });
    }

    try {
      await this.vectorStore.save(vectorStorePath);
      this.logger.log('向量存储已保存');
    } catch (error) {
      this.logger.error('保存向量存储失败:', error);
    }
  }

  /**
   * 加载知识文档元数据
   */
  private async loadKnowledgeDocuments(): Promise<void> {
    const documentsPath = './data/knowledge/documents.json';
    
    if (fs.existsSync(documentsPath)) {
      try {
        const documentsData = fs.readFileSync(documentsPath, 'utf8');
        const documents: KnowledgeDocument[] = JSON.parse(documentsData);
        
        documents.forEach(doc => {
          this.documents.set(doc.id, doc);
        });
        
        this.logger.log(`加载了 ${documents.length} 个知识文档`);
      } catch (error) {
        this.logger.error('加载知识文档失败:', error);
      }
    } else {
      // 如果没有现有文档，加载默认文档
      await this.loadDocumentsFromUrl();
    }
  }

  /**
   * 保存知识文档元数据
   */
  private async saveKnowledgeDocuments(): Promise<void> {
    const documentsPath = './data/knowledge/documents.json';
    const documentsDir = path.dirname(documentsPath);
    
    if (!fs.existsSync(documentsDir)) {
      fs.mkdirSync(documentsDir, { recursive: true });
    }

    try {
      const documents = Array.from(this.documents.values());
      fs.writeFileSync(documentsPath, JSON.stringify(documents, null, 2));
    } catch (error) {
      this.logger.error('保存知识文档失败:', error);
    }
  }

  /**
   * 重建向量索引
   */
  private async rebuildVectorIndex(): Promise<void> {
    this.logger.log('重建向量索引...');
    
    const allDocuments = Array.from(this.documents.values());
    const langchainDocs: Document[] = [];

    for (const doc of allDocuments) {
      const textChunks = await this.textSplitter.splitText(doc.content);
      
      const docChunks = textChunks.map(chunk => new Document({
        pageContent: chunk,
        metadata: {
          documentId: doc.id,
          title: doc.title,
          category: doc.category,
          url: doc.url,
          ...doc.metadata
        }
      }));
      
      langchainDocs.push(...docChunks);
    }

    // 重新创建向量存储
    if (langchainDocs.length > 0) {
      this.vectorStore = await FaissStore.fromDocuments(langchainDocs, this.embeddings);
      await this.saveVectorStore();
    }

    this.logger.log('向量索引重建完成');
  }

  /**
   * 生成文档ID
   */
  private generateDocumentId(): string {
    return `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}