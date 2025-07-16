import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ModelTrainer } from './model-trainer';
import { TrainingController } from './training.controller';

@Module({
  imports: [ConfigModule],
  providers: [ModelTrainer],
  controllers: [TrainingController],
  exports: [ModelTrainer]
})
export class TrainingModule {}