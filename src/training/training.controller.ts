import { Controller, Post, Get, Body, Param, Logger } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { ModelTrainer, TrainingData } from './model-trainer';

@ApiTags('模型训练')
@Controller('training')
export class TrainingController {
  private readonly logger = new Logger(TrainingController.name);
  
  constructor(private readonly modelTrainer: ModelTrainer) {}

  @Post('start')
  @ApiOperation({ summary: '开始模型训练' })
  @ApiResponse({ status: 200, description: '训练开始成功' })
  async startTraining(): Promise<{ message: string; modelPath: string }> {
    this.logger.log('收到模型训练请求');
    
    try {
      const modelPath = await this.modelTrainer.trainModel();
      
      return {
        message: '模型训练完成',
        modelPath
      };
    } catch (error) {
      this.logger.error('训练失败:', error);
      throw error;
    }
  }

  @Get('status/:modelPath')
  @ApiOperation({ summary: '获取模型状态' })
  @ApiResponse({ status: 200, description: '获取状态成功' })
  async getModelStatus(@Param('modelPath') modelPath: string): Promise<any> {
    try {
      const modelInfo = await this.modelTrainer.loadTrainedModel(modelPath);
      return modelInfo;
    } catch (error) {
      this.logger.error('获取模型状态失败:', error);
      throw error;
    }
  }

  @Post('validate/:modelPath')
  @ApiOperation({ summary: '验证模型效果' })
  @ApiResponse({ status: 200, description: '验证完成' })
  async validateModel(
    @Param('modelPath') modelPath: string,
    @Body() testData: TrainingData[]
  ): Promise<any> {
    try {
      const results = await this.modelTrainer.validateModel(modelPath, testData);
      return results;
    } catch (error) {
      this.logger.error('模型验证失败:', error);
      throw error;
    }
  }
}