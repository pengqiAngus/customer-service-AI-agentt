import { NestFactory } from '@nestjs/core';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { ValidationPipe, Logger } from '@nestjs/common';
import { AppModule } from './app.module';

async function bootstrap() {
  const logger = new Logger('Bootstrap');
  
  const app = await NestFactory.create(AppModule);
  
  // 启用CORS
  app.enableCors({
    origin: true,
    credentials: true,
  });

  // 全局验证管道
  app.useGlobalPipes(new ValidationPipe({
    whitelist: true,
    forbidNonWhitelisted: true,
    transform: true,
  }));

  // 设置全局前缀
  app.setGlobalPrefix('api');

  // Swagger文档配置
  const config = new DocumentBuilder()
    .setTitle('AI客户助手系统')
    .setDescription('基于langchain.js的AI客户助手，集成RAG、Memory、Text2SQL功能')
    .setVersion('1.0.0')
    .addTag('模型训练', 'LoRA微调训练相关接口')
    .addTag('AI客户助手', 'AI Agent对话和管理接口')
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('docs', app, document);

  const port = process.env.PORT || 3000;
  await app.listen(port);

  logger.log(`🚀 应用已启动`);
  logger.log(`🌐 服务地址: http://localhost:${port}`);
  logger.log(`📚 API文档: http://localhost:${port}/docs`);
  logger.log(`🎯 API接口: http://localhost:${port}/api`);
}

bootstrap().catch(error => {
  console.error('应用启动失败:', error);
  process.exit(1);
});