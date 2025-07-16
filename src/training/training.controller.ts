import { Controller, Post, Get, Body, Param, Logger } from "@nestjs/common";
import { ApiTags, ApiOperation, ApiResponse } from "@nestjs/swagger";
import { ModelTrainer, TrainingData } from "./model-trainer";
import * as fs from "fs";
import * as path from "path";

@ApiTags("模型训练")
@Controller("training")
export class TrainingController {
  private readonly logger = new Logger(TrainingController.name);

  constructor(private readonly modelTrainer: ModelTrainer) {}

  @Post("start")
  @ApiOperation({ summary: "开始模型训练" })
  @ApiResponse({ status: 200, description: "训练开始成功" })
  async startTraining(): Promise<{ message: string; modelPath: string }> {
    this.logger.log("收到模型训练请求");

    try {
      const modelPath = await this.modelTrainer.trainModel();

      return {
        message: "模型训练完成",
        modelPath,
      };
    } catch (error) {
      this.logger.error("训练失败:", error);
      throw error;
    }
  }

  @Get("status/:modelPath")
  @ApiOperation({ summary: "获取模型状态" })
  @ApiResponse({ status: 200, description: "获取状态成功" })
  async getModelStatus(@Param("modelPath") modelPath: string): Promise<any> {
    try {
      let resolvedModelPath = modelPath;
      if (modelPath === "0" || modelPath === "1") {
        const modelsDir = path.resolve(__dirname, "../../models");
        let modelName =
          modelPath === "0"
            ? "DeepSeek-R1-Distill-Qwen-1.5B"
            : "lora-fine-tuned";
        const modelPathFull = path.join(modelsDir, modelName);
        if (!fs.existsSync(modelPathFull)) {
          throw new Error(`${modelName} 模型未找到`);
        }
        resolvedModelPath = modelPathFull;
      }
      const modelInfo =
        await this.modelTrainer.loadTrainedModel(resolvedModelPath);
      return modelInfo;
    } catch (error) {
      this.logger.error("获取模型状态失败:", error);
      throw error;
    }
  }

  @Post("validate/:modelPath")
  @ApiOperation({ summary: "验证模型效果" })
  @ApiResponse({ status: 200, description: "验证完成" })
  async validateModel(
    @Param("modelPath") modelPath: string,
    @Body() testData: TrainingData[]
  ): Promise<any> {
    try {
      let resolvedModelPath = modelPath;
      if (modelPath === "0" || modelPath === "1") {
        const modelsDir = path.resolve(__dirname, "../../models");
        let modelName =
          modelPath === "0"
            ? "DeepSeek-R1-Distill-Qwen-1.5B"
            : "lora-fine-tuned";
        const modelPathFull = path.join(modelsDir, modelName);
        if (!fs.existsSync(modelPathFull)) {
          throw new Error(`${modelName} 模型未找到`);
        }
        resolvedModelPath = modelPathFull;
      }
      const results = await this.modelTrainer.validateModel(
        resolvedModelPath,
        testData
      );
      return results;
    } catch (error) {
      this.logger.error("模型验证失败:", error);
      throw error;
    }
  }
}
