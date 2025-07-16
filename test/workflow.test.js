const axios = require('axios');

const API_BASE_URL = 'http://localhost:3000/api';

async function testWorkflow() {
  console.log('🧪 测试LangGraph工作流功能...\n');

  const testCases = [
    {
      name: 'SQL查询测试',
      message: '查询所有用户信息',
      expectedPath: ['intent_analysis', 'sql_processing', 'response_generation']
    },
    {
      name: '知识库查询测试',
      message: '产品保修期是多长时间？',
      expectedPath: ['intent_analysis', 'knowledge_processing', 'response_generation']
    },
    {
      name: '常规对话测试',
      message: '你好，我想了解一下你们的服务',
      expectedPath: ['intent_analysis', 'general_conversation', 'response_generation']
    }
  ];

  for (const testCase of testCases) {
    console.log(`📝 测试: ${testCase.name}`);
    console.log(`💬 消息: ${testCase.message}`);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/agent/chat`, {
        message: testCase.message,
        sessionId: `test_session_${Date.now()}`,
        userId: 'test_user'
      });

      console.log(`✅ 响应: ${response.data.response.substring(0, 100)}...`);
      console.log(`🛤️ 实际路径: ${response.data.metadata.workflowPath.join(' -> ')}`);
      console.log(`🎯 意图: ${response.data.metadata.intent} (置信度: ${response.data.metadata.confidence})`);
      console.log(`⏱️ 执行时间: ${response.data.metadata.executionTime}ms`);
      
      // 验证路径
      const actualPath = response.data.metadata.workflowPath;
      const isPathCorrect = JSON.stringify(actualPath) === JSON.stringify(testCase.expectedPath);
      
      if (isPathCorrect) {
        console.log('✅ 路径验证通过');
      } else {
        console.log('❌ 路径验证失败');
        console.log(`   期望: ${testCase.expectedPath.join(' -> ')}`);
        console.log(`   实际: ${actualPath.join(' -> ')}`);
      }
      
      console.log('---\n');
      
    } catch (error) {
      console.log(`❌ 测试失败: ${error.message}\n`);
    }
  }

  // 测试工作流统计
  try {
    console.log('📊 测试工作流统计...');
    const statsResponse = await axios.get(`${API_BASE_URL}/agent/workflow/stats`);
    console.log('✅ 工作流统计:');
    console.log(`   节点数: ${statsResponse.data.nodes.length}`);
    console.log(`   边数: ${statsResponse.data.edges.length}`);
    console.log(`   描述: ${statsResponse.data.description}`);
  } catch (error) {
    console.log(`❌ 统计测试失败: ${error.message}`);
  }
}

// 运行测试
if (require.main === module) {
  testWorkflow().catch(error => {
    console.error('测试执行失败:', error.message);
    process.exit(1);
  });
}

module.exports = { testWorkflow }; 