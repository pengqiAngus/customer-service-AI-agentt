import { Injectable, Logger } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { SqlDatabase } from "langchain/sql_db";
import { Pool } from "pg";

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
  private pool: Pool;
  private schemas: Map<string, TableSchema> = new Map();

  constructor(private configService: ConfigService) {
    this.llm = new ChatOpenAI({
      openAIApiKey: this.configService.get<string>("OPENAI_API_KEY"),
      modelName:
        this.configService.get<string>("OPENAI_MODEL") || "gpt-3.5-turbo",
      temperature: 0,
    });

    this.initializeDatabase();
  }

  /**
   * 初始化数据库连接和表结构
   */
  private async initializeDatabase(): Promise<void> {
    this.logger.log("初始化PostgreSQL连接...");
    try {
      this.pool = new Pool({
        host: this.configService.get<string>("PG_HOST"),
        port: this.configService.get<number>("PG_PORT"),
        user: this.configService.get<string>("PG_USER"),
        password: this.configService.get<string>("PG_PASSWORD"),
        database: this.configService.get<string>("PG_DATABASE"),
      });
      await this.loadTableSchemas();
      this.logger.log("PostgreSQL连接初始化完成");
    } catch (error) {
      this.logger.error("PostgreSQL连接初始化失败:", error);
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
      const explanation = await this.generateExplanation(
        question,
        sqlQuery,
        results
      );

      const executionTime = Date.now() - startTime;

      this.logger.log(`查询完成，耗时: ${executionTime}ms`);

      return {
        query: sqlQuery,
        results,
        explanation,
        executionTime,
      };
    } catch (error) {
      this.logger.error("自然语言转SQL失败:", error);
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
    `) as any; // 用any断言兼容类型

    const tableSchemas = this.generateTableSchemasText();

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      tableSchemas,
      question,
    });

    let sqlQuery = result.text.trim();

    // 清理SQL查询
    sqlQuery = sqlQuery
      .replace(/```sql/g, "")
      .replace(/```/g, "")
      .trim();

    this.logger.log(`生成的SQL查询: ${sqlQuery}`);
    return sqlQuery;
  }

  /**
   * 执行SQL查询
   */
  private async executeSQL(sqlQuery: string): Promise<any[]> {
    // 验证SQL安全性
    if (!this.isQuerySafe(sqlQuery)) {
      throw new Error("检测到不安全的SQL查询");
    }
    const client = await this.pool.connect();
    try {
      const res = await client.query(sqlQuery);
      return res.rows;
    } catch (err) {
      this.logger.error("SQL执行错误:", err);
      throw err;
    } finally {
      client.release();
    }
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
    `) as any; // 用any断言兼容类型

    const chain = new LLMChain({
      llm: this.llm,
      prompt: prompt,
    });

    const result = await chain.call({
      question,
      sqlQuery,
      resultCount: results.length,
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
      "drop",
      "delete",
      "truncate",
      "insert",
      "update",
      "alter",
      "create",
      "exec",
      "execute",
      "sp_",
      "xp_",
      "--",
      "/*",
      "*/",
      ";",
    ];

    for (const keyword of dangerousKeywords) {
      if (query.includes(keyword)) {
        this.logger.warn(`检测到危险关键词: ${keyword}`);
        return false;
      }
    }

    // 只允许SELECT查询
    if (!query.trim().startsWith("select")) {
      this.logger.warn("只允许SELECT查询");
      return false;
    }

    return true;
  }

  /**
   * 加载表结构信息
   */
  private async loadTableSchemas(): Promise<void> {
    // 动态获取PostgreSQL所有表和字段信息
    const client = await this.pool.connect();
    try {
      const tablesRes = await client.query(`
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
      `);
      for (const row of tablesRes.rows) {
        const tableName = row.table_name;
        const columnsRes = await client.query(
          `SELECT column_name, data_type, is_nullable
           FROM information_schema.columns
           WHERE table_name = $1`,
          [tableName]
        );
        const columns: ColumnInfo[] = columnsRes.rows.map((col) => ({
          name: col.column_name,
          type: col.data_type,
          nullable: col.is_nullable === "YES",
          description: "",
        }));
        this.schemas.set(tableName, {
          name: tableName,
          columns,
          description: "",
        });
      }
      this.logger.log("表结构信息加载完成");
    } finally {
      client.release();
    }
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

    return schemaTexts.join("\n");
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
      throw new Error("不安全的SQL查询");
    }

    return this.executeSQL(sqlQuery);
  }
}
