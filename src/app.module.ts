import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { TrainingModule } from './training/training.module';
import { AgentModule } from './agent/agent.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    TrainingModule,
    AgentModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}