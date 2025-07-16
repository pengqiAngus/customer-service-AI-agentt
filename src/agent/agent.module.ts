import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AIAgentService } from './ai-agent.service';
import { ConversationMemory } from './memory/conversation-memory';
import { KnowledgeBase } from './rag/knowledge-base';
import { Text2SQL } from './sql/text2sql';
import { AgentController } from './agent.controller';

@Module({
  imports: [ConfigModule],
  providers: [
    AIAgentService,
    ConversationMemory,
    KnowledgeBase,
    Text2SQL,
  ],
  controllers: [AgentController],
  exports: [
    AIAgentService,
    ConversationMemory,
    KnowledgeBase,
    Text2SQL,
  ]
})
export class AgentModule {}