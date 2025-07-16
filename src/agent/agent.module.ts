import { Module } from "@nestjs/common";
import { ConfigModule } from "@nestjs/config";
import { AIAgentService } from "./ai-agent.service";
import { ConversationMemory } from "./memory/conversation-memory";
import { KnowledgeBase } from "./rag/knowledge-base";
import { Text2SQL } from "./sql/text2sql";
import { AgentWorkflow } from "./workflow/agent-workflow";
import { AgentController } from "./agent.controller";

@Module({
  imports: [ConfigModule],
  providers: [
    AIAgentService,
    ConversationMemory,
    KnowledgeBase,
    ...(process.env.TEXT2SQL_ENABLED === "true" ? [Text2SQL] : []),
    AgentWorkflow,
  ],
  controllers: [AgentController],
  exports: [
    AIAgentService,
    ConversationMemory,
    KnowledgeBase,
    ...(process.env.TEXT2SQL_ENABLED === "true" ? [Text2SQL] : []),
    AgentWorkflow,
  ],
})
export class AgentModule {}
