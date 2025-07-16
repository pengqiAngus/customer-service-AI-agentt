const axios = require('axios');

const API_BASE_URL = 'http://localhost:3000/api';

class AIAgentDemo {
  constructor() {
    this.sessionId = `demo_session_${Date.now()}`;
    console.log('🎯 AI客户助手演示程序');
    console.log(`📱 会话ID: ${this.sessionId}`);
    console.log('=' * 50);
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(method, endpoint, data = null) {
    try {
      const config = {
        method,
        url: `${API_BASE_URL}${endpoint}`,
        headers: {
          'Content-Type': 'application/json',
        }
      };

      if (data) {
        config.data = data;
      }

      const response = await axios(config);
      return response.data;
    } catch (error) {
      console.error(`❌ 请求失败: ${error.message}`);
      if (error.response) {
        console.error(`状态码: ${error.response.status}`);
        console.error(`错误详情: ${JSON.stringify(error.response.data, null, 2)}`);
      }
      throw error;
    }
  }

  async healthCheck() {
    console.log('\n🔍 正在检查服务状态...');
    try {
      const health = await this.makeRequest('GET', '/agent/health');
      console.log('✅ 服务状态正常');
      console.log(`📊 服务信息:`, health);
      return true;
    } catch (error) {
      console.error('❌ 服务不可用，请确保服务已启动');
      return false;
    }
  }

  async demonstrateTraining() {
    console.log('\n🎓 演示模型训练功能...');
    
    try {
      console.log('📚 开始模型训练...');
      const trainingResult = await this.makeRequest('POST', '/training/start');
      console.log('✅ 训练请求已提交:', trainingResult);
      
      await this.delay(2000);
      
      // 注意：实际项目中这里会是真实的模型路径
      const modelPath = './models/lora-fine-tuned';
      console.log(`📋 获取模型状态: ${modelPath}`);
      
      // 这里会尝试获取模型状态，如果模型不存在会报错，这是正常的
      try {
        const status = await this.makeRequest('GET', `/training/status/${encodeURIComponent(modelPath)}`);
        console.log('📈 模型状态:', status);
      } catch (error) {
        console.log('ℹ️  模型尚未完成训练或路径不存在（演示环境正常现象）');
      }
      
    } catch (error) {
      console.log('ℹ️  训练功能演示完成（演示环境可能出现错误是正常的）');
    }
  }

  async demonstrateKnowledgeBase() {
    console.log('\n📚 演示知识库功能...');
    
    try {
      // 获取所有文档
      console.log('📋 获取知识库文档列表...');
      const documents = await this.makeRequest('GET', '/agent/knowledge/documents');
      console.log(`📖 当前知识库包含 ${documents.documents.length} 个文档`);
      
      if (documents.documents.length > 0) {
        console.log('📑 文档列表:');
        documents.documents.forEach((doc, index) => {
          console.log(`  ${index + 1}. ${doc.title} (${doc.category})`);
        });
      }

      // 搜索知识库
      console.log('\n🔍 搜索知识库: "保修"');
      const searchResults = await this.makeRequest('POST', '/agent/knowledge/search', {
        query: '保修',
        limit: 3
      });
      
      console.log(`📝 找到 ${searchResults.results.length} 个相关结果:`);
      searchResults.results.forEach((result, index) => {
        console.log(`\n${index + 1}. ${result.document.title}`);
        console.log(`   相关性: ${(result.score * 100).toFixed(1)}%`);
        console.log(`   内容摘要: ${result.relevantChunk.substring(0, 100)}...`);
      });

    } catch (error) {
      console.log('ℹ️  知识库功能演示遇到问题，可能需要初始化');
    }
  }

  async demonstrateText2SQL() {
    console.log('\n🗄️ 演示Text2SQL功能...');
    
    try {
      // 获取数据库结构
      console.log('📊 获取数据库表结构...');
      const schemas = await this.makeRequest('GET', '/agent/sql/schemas');
      console.log(`📋 数据库包含 ${schemas.schemas.length} 个表:`);
      schemas.schemas.forEach(schema => {
        console.log(`  - ${schema.name}: ${schema.description}`);
      });

      // 自然语言查询
      console.log('\n💬 自然语言查询: "查询所有用户信息"');
      const queryResult = await this.makeRequest('POST', '/agent/sql/query', {
        question: '查询所有用户信息'
      });
      
      console.log(`🔍 生成的SQL: ${queryResult.query}`);
      console.log(`📊 查询结果数量: ${queryResult.results.length}`);
      console.log(`📝 查询解释: ${queryResult.explanation}`);
      console.log(`⏱️ 执行时间: ${queryResult.executionTime}ms`);
      
      if (queryResult.results.length > 0) {
        console.log('\n📋 查询结果预览:');
        console.log(JSON.stringify(queryResult.results.slice(0, 2), null, 2));
      }

    } catch (error) {
      console.log('ℹ️  Text2SQL功能演示遇到问题，可能需要数据库初始化');
    }
  }

  async demonstrateConversation() {
    console.log('\n💬 演示AI Agent对话功能...');
    
    const conversations = [
      '你好，我是新用户，想了解一下你们的服务',
      '我想查询我的订单状态',
      '产品保修期是多长时间？',
      '如果产品出现故障怎么办？',
      '谢谢你的帮助！'
    ];

    for (const [index, message] of conversations.entries()) {
      console.log(`\n👤 用户 (${index + 1}/5): ${message}`);
      
      try {
        const response = await this.makeRequest('POST', '/agent/chat', {
          message: message,
          sessionId: this.sessionId,
          userId: 'demo_user'
        });

        console.log(`🤖 AI助手: ${response.response}`);
        console.log(`📊 元数据:`);
        console.log(`   - 使用记忆: ${response.metadata.usedMemory}`);
        console.log(`   - 使用RAG: ${response.metadata.usedRAG}`);
        console.log(`   - 使用SQL: ${response.metadata.usedSQL}`);
        console.log(`   - 执行时间: ${response.metadata.executionTime}ms`);
        
        if (response.sources && response.sources.length > 0) {
          console.log(`   - 信息来源: ${response.sources.join(', ')}`);
        }

        await this.delay(1000); // 模拟真实对话间隔
        
      } catch (error) {
        console.log(`❌ 对话失败: ${error.message}`);
        break;
      }
    }
  }

  async demonstrateSessionAnalysis() {
    console.log('\n📈 演示会话分析功能...');
    
    try {
      const analysis = await this.makeRequest('GET', `/agent/session/${this.sessionId}/analysis`);
      
      console.log('🔍 会话分析结果:');
      console.log(`   - 情感倾向: ${analysis.sentiment}`);
      console.log(`   - 用户意图: ${analysis.intent}`);
      console.log(`   - 讨论话题: ${analysis.topics.join(', ')}`);
      console.log(`   - 总消息数: ${analysis.totalMessages}`);

      // 搜索历史对话
      console.log('\n🔍 搜索历史对话: "订单"');
      const searchResults = await this.makeRequest('GET', `/agent/session/${this.sessionId}/search`, {
        params: { query: '订单' }
      });
      
      if (searchResults.length > 0) {
        console.log(`📝 找到 ${searchResults.length} 条相关历史消息`);
      } else {
        console.log('📝 未找到相关历史消息');
      }

    } catch (error) {
      console.log('ℹ️  会话分析功能演示遇到问题');
    }
  }

  async demonstrateStats() {
    console.log('\n📊 演示系统统计功能...');
    
    try {
      const stats = await this.makeRequest('GET', '/agent/stats');
      
      console.log('📈 系统统计信息:');
      console.log(`   - 活跃会话数: ${stats.activeSessions}`);
      console.log(`   - 知识库文档数: ${stats.totalDocuments}`);
      console.log(`   - 数据库表数: ${stats.databaseTables}`);
      console.log(`   - 统计时间: ${stats.timestamp}`);
      
      if (stats.documentsByCategory) {
        console.log('   - 文档分类统计:');
        Object.entries(stats.documentsByCategory).forEach(([category, count]) => {
          console.log(`     * ${category}: ${count}个文档`);
        });
      }

    } catch (error) {
      console.log('ℹ️  统计功能演示遇到问题');
    }
  }

  async cleanup() {
    console.log('\n🧹 清理演示环境...');
    
    try {
      await this.makeRequest('DELETE', `/agent/session/${this.sessionId}`);
      console.log('✅ 会话记忆已清除');
    } catch (error) {
      console.log('ℹ️  清理过程中遇到小问题，可以忽略');
    }
  }

  async runFullDemo() {
    console.log('🎬 开始完整功能演示...\n');
    
    // 健康检查
    const isHealthy = await this.healthCheck();
    if (!isHealthy) {
      console.log('\n❌ 服务不可用，请先启动服务:');
      console.log('   pnpm run start:dev');
      return;
    }

    await this.delay(1000);

    // 演示各个功能模块
    await this.demonstrateTraining();
    await this.demonstrateKnowledgeBase();
    await this.demonstrateText2SQL();
    await this.demonstrateConversation();
    await this.demonstrateSessionAnalysis();
    await this.demonstrateStats();
    
    // 清理
    await this.cleanup();

    console.log('\n🎉 演示完成！');
    console.log('\n📝 接下来你可以:');
    console.log('1. 访问 http://localhost:3000/docs 查看完整API文档');
    console.log('2. 使用 Postman 或其他工具测试API接口');
    console.log('3. 根据需求修改配置文件和训练数据');
    console.log('4. 扩展知识库内容和数据库结构');
  }
}

// 运行演示
if (require.main === module) {
  const demo = new AIAgentDemo();
  demo.runFullDemo().catch(error => {
    console.error('\n❌ 演示过程中发生错误:', error.message);
    process.exit(1);
  });
}

module.exports = AIAgentDemo;