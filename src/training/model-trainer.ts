import { Injectable, Logger } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import axios from "axios";
import * as fs from "fs";
import * as path from "path";

export interface TrainingData {
  instruction: string;
  content: string;
  summary: string;
}

export interface LoRAConfig {
  rank: number;
  alpha: number;
  dropout: number;
  targetModules: string[];
  taskType: string;
}

@Injectable()
export class ModelTrainer {
  private readonly logger = new Logger(ModelTrainer.name);

  constructor(private configService: ConfigService) {}

  /**
   * 第一步：模型训练微调
   * 使用LoRA (Low-Rank Adaptation) 进行高效微调
   */
  async trainModel(): Promise<string> {
    this.logger.log("开始模型训练微调过程...");

    try {
      // 1. 下载训练数据集
      const trainingData = await this.downloadTrainingData();

      // 2. 预处理数据
      const processedData = await this.preprocessData(trainingData);

      // 3. 配置LoRA参数
      const loraConfig = this.getLoRAConfig();

      // 4. 开始微调训练
      const modelPath = await this.startFineTuning(processedData, loraConfig);

      this.logger.log(`模型训练完成，保存路径: ${modelPath}`);
      return modelPath;
    } catch (error) {
      this.logger.error("模型训练失败:", error);
      throw error;
    }
  }

  /**
   * 从本地目录读取所有包含 'train' 的 json 文件作为训练数据，目录路径从 .env 的 TRAINING_DATA_URL 字段获取
   */
  private async downloadTrainingData(): Promise<TrainingData[]> {
    const dataDir = this.configService.get<string>("TRAINING_DATA_URL");
    this.logger.log(`从本地目录 ${dataDir} 读取训练数据...`);
    try {
      const files = fs
        .readdirSync(dataDir)
        .filter((f) => f.endsWith(".json") && f.includes("train"));
      if (files.length === 0) {
        throw new Error("未找到包含 train 的 json 文件");
      }
      let allData: TrainingData[] = [];
      for (const file of files) {
        const filePath = path.join(dataDir, file);
        const fileContent = fs.readFileSync(filePath, "utf8");
        // 按行读取，每行一个对象
        const lines = fileContent
          .split(/\r?\n/)
          .filter((line) => line.trim() !== "");
        for (const line of lines) {
          try {
            const jsonData = JSON.parse(line);
            allData.push(jsonData as TrainingData);
          } catch (e) {
            this.logger.warn(`文件${file}中有无法解析的行: ${line}`);
          }
        }
      }
      this.logger.log(`成功读取 ${allData.length} 条训练数据`);
      return allData;
    } catch (error) {
      this.logger.error("读取本地训练数据失败:", error);
      throw new Error(`无法从本地目录 ${dataDir} 读取训练数据`);
    }
  }

  /**
   * 预处理训练数据
   */
  private async preprocessData(rawData: any[]): Promise<any[]> {
    this.logger.log("预处理训练数据...");
    const processedData = rawData.map((item, index) => {
      // 兼容不同数据格式
      let instruction = "请根据商品属性生成商品文案";
      let input = item.content || "";
      let output = item.summary || "";
      return {
        id: index,
        text: `### 指令:\n${instruction}\n\n### 输入:\n${input}\n\n### 回复:\n${output}`,
        instruction,
        input,
        output,
        length: output.length,
      };
    });
    // 保存预处理后的数据
    const dataDir = "./data/training";
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    fs.writeFileSync(
      path.join(dataDir, "processed_training_data.json"),
      JSON.stringify(processedData, null, 2)
    );
    this.logger.log(`预处理完成，生成 ${processedData.length} 条训练样本`);
    return processedData;
  }

  /**
   * 获取LoRA配置参数
   */
  private getLoRAConfig(): LoRAConfig {
    return {
      rank: 16, // LoRA的秩，控制适配器的大小
      alpha: 32, // LoRA的缩放参数
      dropout: 0.1, // Dropout率
      targetModules: [
        // 目标模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
      ],
      taskType: "CAUSAL_LM", // 因果语言模型
    };
  }

  /**
   * 开始LoRA微调训练
   */
  private async startFineTuning(
    trainingData: any[],
    loraConfig: LoRAConfig
  ): Promise<string> {
    this.logger.log("开始LoRA微调训练...");

    const baseModelUrl = this.configService.get<string>("BASE_MODEL_URL");
    const modelOutputPath = this.configService.get<string>("LORA_MODEL_PATH");

    // 创建输出目录
    if (!fs.existsSync(modelOutputPath)) {
      fs.mkdirSync(modelOutputPath, { recursive: true });
    }

    // 模拟训练过程（实际应用中会调用真实的微调API或本地训练脚本）
    this.logger.log("配置训练参数...");
    const trainingConfig = {
      baseModel: baseModelUrl,
      loraConfig: loraConfig,
      trainingData: trainingData,
      hyperparameters: {
        learningRate: 2e-4,
        batchSize: 4,
        numEpochs: 3,
        warmupSteps: 100,
        maxSeqLength: 512,
        gradientAccumulationSteps: 4,
      },
      outputDir: modelOutputPath,
    };

    // 保存训练配置
    fs.writeFileSync(
      path.join(modelOutputPath, "training_config.json"),
      JSON.stringify(trainingConfig, null, 2)
    );

    // 模拟训练进度
    for (let epoch = 1; epoch <= 3; epoch++) {
      this.logger.log(`训练 Epoch ${epoch}/3...`);
      await this.sleep(2000); // 模拟训练时间

      const loss = (0.5 - epoch * 0.1).toFixed(4);
      this.logger.log(`Epoch ${epoch} 完成, Loss: ${loss}`);
    }

    // 保存微调后的模型信息
    const modelInfo = {
      modelType: "LoRA-FineTuned",
      baseModel: baseModelUrl,
      trainingDate: new Date().toISOString(),
      trainingDataSize: trainingData.length,
      loraConfig: loraConfig,
      modelPath: modelOutputPath,
      status: "completed",
    };

    fs.writeFileSync(
      path.join(modelOutputPath, "model_info.json"),
      JSON.stringify(modelInfo, null, 2)
    );

    this.logger.log("LoRA微调训练完成！");
    return modelOutputPath;
  }

  /**
   * 加载已训练的模型
   */
  async loadTrainedModel(modelPath: string): Promise<any> {
    this.logger.log(`加载微调模型: ${modelPath}`);

    if (!fs.existsSync(modelPath)) {
      throw new Error(`模型路径不存在: ${modelPath}`);
    }

    const modelInfoPath = path.join(modelPath, "model_info.json");
    if (!fs.existsSync(modelInfoPath)) {
      throw new Error("模型信息文件不存在");
    }

    const modelInfo = JSON.parse(fs.readFileSync(modelInfoPath, "utf8"));
    this.logger.log("模型加载成功:", modelInfo.modelType);

    return {
      modelInfo,
      isLoaded: true,
      loadTime: new Date().toISOString(),
    };
  }

  /**
   * 验证微调效果
   */
  async validateModel(
    modelPath: string,
    testData: TrainingData[]
  ): Promise<any> {
    this.logger.log("开始模型验证...");
    const model = await this.loadTrainedModel(modelPath);
    const results = [];
    for (const testCase of testData) {
      // 兼容不同数据格式
      const instruction = testCase.instruction || "请根据商品属性生成商品文案";
      const input = testCase.content || "";
      const expected = testCase.summary || "";
      // 模拟推理过程
      const prediction = await this.predict(model, instruction, input);
      results.push({
        input,
        expected,
        predicted: prediction,
        score: this.calculateSimilarity(expected, prediction),
      });
    }
    const averageScore =
      results.reduce((sum, r) => sum + r.score, 0) / results.length;
    this.logger.log(`模型验证完成，平均得分: ${averageScore.toFixed(2)}`);
    return {
      averageScore,
      results,
      totalTests: results.length,
    };
  }

  private async predict(
    model: any,
    instruction: string,
    input: string
  ): Promise<string> {
    // 简单模拟：直接返回 input 或“根据属性生成文案：”+input
    return `根据属性生成文案：${input}`;
  }

  private calculateSimilarity(expected: string, predicted: string): number {
    // 简单的相似度计算（实际应用中可使用更sophisticated的方法）
    const expectedWords = expected.toLowerCase().split(" ");
    const predictedWords = predicted.toLowerCase().split(" ");

    const intersection = expectedWords.filter((word) =>
      predictedWords.includes(word)
    );
    return (
      intersection.length /
      Math.max(expectedWords.length, predictedWords.length)
    );
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
