import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { LLMChain } from 'langchain/chains';
import { SqlDatabase } from 'langchain/sql_db';
import { Database } from 'sqlite3';
import * as path from 'path';
import * as fs from 'fs';

export interface SQLQueryResult {
  query: string;
  results: any[];
  explanation: string;
  executionTime: number;
}

export interface TableSchema {
  name: string;
  columns: ColumnInfo[];
  description: string;
}

export interface ColumnInfo {
  name: string;
  type: string;
  nullable: boolean;
  description: string;
}

@Injectable()
export class Text2SQL {
  private readonly logger = new Logger(Text2SQL.name);
  private llm: ChatOpenAI;
  private sqlDatabase: SqlDatabase;
  private db: Database;
  private schemas: Map<string, TableSchema> = new Map();

  constructor(private configService: ConfigService) {
    this.llm = new ChatOpenAI({
      openAIApiKey: this.configService.get<string>('OPENAI_API_KEY'),
      modelName: this.configService.get<string>('OPENAI_MODEL') || 'gpt-3.5-turbo',
      temperature: 0,
    });

    this.initializeDatabase();
  }

  /**
   * 初始化数据库连接和表结构
   */
  private async initializeDatabase(): Promise<void> {
    this.logger.log('初始化数据库连接...');

    try {
      // 创建SQLite数据库连接
      const dbPath = './data/customer_service.db';
      const dbDir = path.dirname(dbPath);
      
      if (!fs.existsSync(dbDir)) {
        fs.mkdirSync(dbDir, { recursive: true });
      }

      this.db = new Database(dbPath);
      
      // 创建示例表结构
      await this.createTables();
      await this.loadSampleData();
      await this.loadTableSchemas();

      // 初始化LangChain SQL数据库
      this.sqlDatabase = await SqlDatabase.fromDataSourceParams({
        appDataSource: {
          type: 'sqlite',
          database: dbPath,
        }
      });

      this.logger.log('数据库初始化完成');
    } catch (error) {
      this.logger.error('数据库初始化失败:', error);
      throw error;
    }
  }

  /**
   * 将自然语言转换为SQL查询
   */
  async naturalLanguageToSQL(question: string): Promise<SQLQueryResult> {
    const startTime = Date.now();
    
    try {
      this.logger.log(`处理自然语言查询: ${question}`);

      // 1. 分析问题并生成SQL
      const sqlQuery = await this.generateSQL(question);
      
      // 2. 执行SQL查询
      const results = await this.executeSQL(sqlQuery);
      
      // 3. 生成解释
      const explanation = await this.generateExplanation(question, sqlQuery, results);
      
      const executionTime = Date.now() - startTime;
      
      this.logger.log(`查询完成，耗时: ${executionTime}ms`);
      
      return {
        query: sqlQuery,
        results,
        explanation,
        executionTime
      };
      
    } catch (error) {
      this.logger.error('自然语言转SQL失败:', error);
      throw error;
    }
  }

  /**
   * 生成SQL查询
   */
  private async generateSQL(question: string): Promise<string> {
    const prompt = PromptTemplate.fromTemplate(`
你是一个专业的SQL专家。根据用户的自然语言问题，生成对应的SQL查询语句。

数据库表结构信息：
{tableSchemas}

用户问题: {question}

请注意：
1. 只返回SQL查询语句，不要包含任何解释
2. 使用标准SQL语法
3. 确保查询是安全的，避免SQL注入
4. 如果问题模糊，选择最合理的解释

SQL查询:
    `);

    const tableSchemas = this.generateTableSchemasText();
    
    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      tableSchemas,
      question
    });

    let sqlQuery = result.text.trim();
    
    // 清理SQL查询
    sqlQuery = sqlQuery.replace(/```sql/g, '').replace(/```/g, '').trim();
    
    this.logger.log(`生成的SQL查询: ${sqlQuery}`);
    return sqlQuery;
  }

  /**
   * 执行SQL查询
   */
  private async executeSQL(sqlQuery: string): Promise<any[]> {
    return new Promise((resolve, reject) => {
      // 验证SQL安全性
      if (!this.isQuerySafe(sqlQuery)) {
        reject(new Error('检测到不安全的SQL查询'));
        return;
      }

      this.db.all(sqlQuery, (err, rows) => {
        if (err) {
          this.logger.error('SQL执行错误:', err);
          reject(err);
        } else {
          this.logger.log(`查询返回 ${rows.length} 条记录`);
          resolve(rows);
        }
      });
    });
  }

  /**
   * 生成查询解释
   */
  private async generateExplanation(
    question: string, 
    sqlQuery: string, 
    results: any[]
  ): Promise<string> {
    const prompt = PromptTemplate.fromTemplate(`
用户问题: {question}
执行的SQL查询: {sqlQuery}
查询结果数量: {resultCount}

请用中文解释这个查询的含义和结果：
1. 解释SQL查询做了什么
2. 总结查询结果
3. 回答用户的原始问题

解释:
    `);

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      question,
      sqlQuery,
      resultCount: results.length
    });

    return result.text.trim();
  }

  /**
   * 验证SQL查询安全性
   */
  private isQuerySafe(sqlQuery: string): boolean {
    const query = sqlQuery.toLowerCase();
    
    // 检查危险关键词
    const dangerousKeywords = [
      'drop', 'delete', 'truncate', 'insert', 'update',
      'alter', 'create', 'exec', 'execute', 'sp_',
      'xp_', '--', '/*', '*/', ';'
    ];

    for (const keyword of dangerousKeywords) {
      if (query.includes(keyword)) {
        this.logger.warn(`检测到危险关键词: ${keyword}`);
        return false;
      }
    }

    // 只允许SELECT查询
    if (!query.trim().startsWith('select')) {
      this.logger.warn('只允许SELECT查询');
      return false;
    }

    return true;
  }

  /**
   * 创建示例表
   */
  private async createTables(): Promise<void> {
    const createTablesSQL = [
      // 用户表
      `CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(50) NOT NULL,
        email VARCHAR(100) NOT NULL,
        phone VARCHAR(20),
        registration_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'active'
      )`,
      
      // 订单表
      `CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        order_number VARCHAR(50) UNIQUE NOT NULL,
        total_amount DECIMAL(10,2),
        status VARCHAR(20) DEFAULT 'pending',
        order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        shipping_address TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
      )`,
      
      // 产品表
      `CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(100) NOT NULL,
        category VARCHAR(50),
        price DECIMAL(10,2),
        stock_quantity INTEGER DEFAULT 0,
        description TEXT,
        warranty_months INTEGER DEFAULT 12
      )`,
      
      // 订单项表
      `CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        unit_price DECIMAL(10,2),
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
      )`,
      
      // 客服工单表
      `CREATE TABLE IF NOT EXISTS support_tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title VARCHAR(200),
        description TEXT,
        status VARCHAR(20) DEFAULT 'open',
        priority VARCHAR(20) DEFAULT 'medium',
        category VARCHAR(50),
        created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        resolved_date DATETIME,
        FOREIGN KEY (user_id) REFERENCES users(id)
      )`
    ];

    for (const sql of createTablesSQL) {
      await new Promise<void>((resolve, reject) => {
        this.db.run(sql, (err) => {
          if (err) reject(err);
          else resolve();
        });
      });
    }

    this.logger.log('数据表创建完成');
  }

  /**
   * 加载示例数据
   */
  private async loadSampleData(): Promise<void> {
    // 检查是否已有数据
    const userCount = await new Promise<number>((resolve) => {
      this.db.get('SELECT COUNT(*) as count FROM users', (err, row: any) => {
        resolve(row ? row.count : 0);
      });
    });

    if (userCount > 0) {
      this.logger.log('数据已存在，跳过示例数据加载');
      return;
    }

    // 插入示例数据
    const sampleData = [
      // 用户数据
      `INSERT INTO users (username, email, phone, status) VALUES 
        ('张三', 'zhangsan@email.com', '13812345678', 'active'),
        ('李四', 'lisi@email.com', '13987654321', 'active'),
        ('王五', 'wangwu@email.com', '13555666777', 'inactive')`,
      
      // 产品数据
      `INSERT INTO products (name, category, price, stock_quantity, warranty_months) VALUES 
        ('智能手机X1', '电子产品', 2999.00, 50, 24),
        ('蓝牙耳机Pro', '配件', 299.00, 100, 12),
        ('笔记本电脑Ultra', '电脑', 8999.00, 20, 36)`,
      
      // 订单数据
      `INSERT INTO orders (user_id, order_number, total_amount, status, shipping_address) VALUES 
        (1, 'ORD20240101001', 3298.00, 'completed', '北京市朝阳区xxx街道'),
        (2, 'ORD20240101002', 8999.00, 'shipping', '上海市浦东新区xxx路'),
        (1, 'ORD20240102001', 299.00, 'pending', '北京市朝阳区xxx街道')`,
      
      // 订单项数据
      `INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES 
        (1, 1, 1, 2999.00),
        (1, 2, 1, 299.00),
        (2, 3, 1, 8999.00),
        (3, 2, 1, 299.00)`,
      
      // 客服工单数据
      `INSERT INTO support_tickets (user_id, title, description, status, priority, category) VALUES 
        (1, '产品无法开机', '购买的智能手机无法正常开机，按电源键没有反应', 'open', 'high', '技术支持'),
        (2, '申请退款', '对购买的笔记本不满意，希望申请退款', 'in_progress', 'medium', '退款申请'),
        (1, '查询物流信息', '想了解我的订单物流状态', 'resolved', 'low', '订单查询')`
    ];

    for (const sql of sampleData) {
      await new Promise<void>((resolve, reject) => {
        this.db.run(sql, (err) => {
          if (err) reject(err);
          else resolve();
        });
      });
    }

    this.logger.log('示例数据加载完成');
  }

  /**
   * 加载表结构信息
   */
  private async loadTableSchemas(): Promise<void> {
    const schemas: TableSchema[] = [
      {
        name: 'users',
        description: '用户信息表',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, description: '用户ID，主键' },
          { name: 'username', type: 'VARCHAR(50)', nullable: false, description: '用户姓名' },
          { name: 'email', type: 'VARCHAR(100)', nullable: false, description: '邮箱地址' },
          { name: 'phone', type: 'VARCHAR(20)', nullable: true, description: '手机号码' },
          { name: 'registration_date', type: 'DATETIME', nullable: true, description: '注册时间' },
          { name: 'status', type: 'VARCHAR(20)', nullable: true, description: '账户状态（active/inactive）' }
        ]
      },
      {
        name: 'orders',
        description: '订单信息表',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, description: '订单ID，主键' },
          { name: 'user_id', type: 'INTEGER', nullable: true, description: '用户ID，外键' },
          { name: 'order_number', type: 'VARCHAR(50)', nullable: false, description: '订单号' },
          { name: 'total_amount', type: 'DECIMAL(10,2)', nullable: true, description: '订单总金额' },
          { name: 'status', type: 'VARCHAR(20)', nullable: true, description: '订单状态（pending/completed/shipping/cancelled）' },
          { name: 'order_date', type: 'DATETIME', nullable: true, description: '下单时间' },
          { name: 'shipping_address', type: 'TEXT', nullable: true, description: '收货地址' }
        ]
      },
      {
        name: 'products',
        description: '产品信息表',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, description: '产品ID，主键' },
          { name: 'name', type: 'VARCHAR(100)', nullable: false, description: '产品名称' },
          { name: 'category', type: 'VARCHAR(50)', nullable: true, description: '产品类别' },
          { name: 'price', type: 'DECIMAL(10,2)', nullable: true, description: '产品价格' },
          { name: 'stock_quantity', type: 'INTEGER', nullable: true, description: '库存数量' },
          { name: 'description', type: 'TEXT', nullable: true, description: '产品描述' },
          { name: 'warranty_months', type: 'INTEGER', nullable: true, description: '保修期（月）' }
        ]
      },
      {
        name: 'support_tickets',
        description: '客服工单表',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, description: '工单ID，主键' },
          { name: 'user_id', type: 'INTEGER', nullable: true, description: '用户ID，外键' },
          { name: 'title', type: 'VARCHAR(200)', nullable: true, description: '工单标题' },
          { name: 'description', type: 'TEXT', nullable: true, description: '问题描述' },
          { name: 'status', type: 'VARCHAR(20)', nullable: true, description: '工单状态（open/in_progress/resolved/closed）' },
          { name: 'priority', type: 'VARCHAR(20)', nullable: true, description: '优先级（low/medium/high/urgent）' },
          { name: 'category', type: 'VARCHAR(50)', nullable: true, description: '工单类别' },
          { name: 'created_date', type: 'DATETIME', nullable: true, description: '创建时间' },
          { name: 'resolved_date', type: 'DATETIME', nullable: true, description: '解决时间' }
        ]
      }
    ];

    schemas.forEach(schema => {
      this.schemas.set(schema.name, schema);
    });

    this.logger.log('表结构信息加载完成');
  }

  /**
   * 生成表结构文本描述
   */
  private generateTableSchemasText(): string {
    const schemaTexts: string[] = [];
    
    for (const schema of this.schemas.values()) {
      let schemaText = `表名: ${schema.name}\n描述: ${schema.description}\n字段:\n`;
      
      for (const column of schema.columns) {
        schemaText += `  - ${column.name} (${column.type}): ${column.description}\n`;
      }
      
      schemaTexts.push(schemaText);
    }
    
    return schemaTexts.join('\n');
  }

  /**
   * 获取表结构信息
   */
  async getTableSchemas(): Promise<TableSchema[]> {
    return Array.from(this.schemas.values());
  }

  /**
   * 直接执行SQL查询（仅用于测试）
   */
  async executeDirectSQL(sqlQuery: string): Promise<any[]> {
    if (!this.isQuerySafe(sqlQuery)) {
      throw new Error('不安全的SQL查询');
    }
    
    return this.executeSQL(sqlQuery);
  }
}