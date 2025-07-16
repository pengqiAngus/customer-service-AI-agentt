import os
import json
import logging
import sqlalchemy
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import create_engine, text, inspect
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

from app.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLOutputParser(BaseOutputParser):
    """从LLM输出解析SQL查询"""
    
    def parse(self, text: str) -> str:
        """从文本中提取SQL查询"""
        # Remove markdown code blocks if present
        if "```sql" in text:
            start = text.find("```sql") + 6
            end = text.find("```", start)
            if end != -1:
                sql = text[start:end].strip()
            else:
                sql = text[start:].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                sql = text[start:end].strip()
            else:
                sql = text[start:].strip()
        else:
            sql = text.strip()
        
        return sql

class Text2SQL:
    """客户服务问题的文本转SQL"""
    
    def __init__(self):
        self.enabled = config.ENABLE_TEXT2SQL
        self.engine = None
        self.llm = None
        self.local_tokenizer = None
        self.local_model = None
        self.local_pipe = None
        self.schema_info = {}
        self.sql_parser = SQLOutputParser()
        
        if self.enabled:
            self._initialize_components()
    
    def _initialize_components(self):
        """初始化数据库连接和LLM"""
        try:
            # Initialize database connection
            if config.DATABASE_URL:
                self.engine = create_engine(config.DATABASE_URL)
                self._load_schema_info()
                logger.info("Database connection established")
            else:
                logger.warning("DATABASE_URL not configured. Text2SQL will be disabled.")
                self.enabled = False
                return
            
            # 加载本地模型用于SQL生成
            try:
                logger.info(f"Loading local model for Text2SQL: {config.MODEL_NAME}")
                self.local_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
                self.local_model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
                self.local_pipe = pipeline(
                    "text-generation",
                    model=self.local_model,
                    tokenizer=self.local_tokenizer,
                    max_new_tokens=256,
                    temperature=0.1,
                    device=0 if hasattr(self.local_model, 'cuda') and self.local_model.device.type == 'cuda' else -1
                )
                logger.info("本地LLM加载完成 (Text2SQL)")
            except Exception as e:
                logger.error(f"本地模型加载失败 (Text2SQL): {e}")
                self.local_pipe = None
            
        except Exception as e:
            logger.error(f"Error initializing Text2SQL components: {e}")
            self.enabled = False
    
    def _load_schema_info(self):
        """加载数据库结构信息"""
        try:
            # Try to load from config file first
            if os.path.exists(config.SQL_TABLES_INFO_PATH):
                with open(config.SQL_TABLES_INFO_PATH, 'r', encoding='utf-8') as f:
                    self.schema_info = json.load(f)
                logger.info("Loaded schema info from config file")
            else:
                # Inspect database schema
                self._inspect_database_schema()
                self._create_sample_schema_info()
        
        except Exception as e:
            logger.error(f"Error loading schema info: {e}")
            self._create_sample_schema_info()
    
    def _inspect_database_schema(self):
        """检查数据库以获取结构信息"""
        try:
            if not self.engine:
                return
            
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            schema_info = {"tables": {}}
            
            for table_name in tables:
                columns = inspector.get_columns(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                indexes = inspector.get_indexes(table_name)
                
                table_info = {
                    "description": f"表: {table_name}",
                    "columns": [],
                    "foreign_keys": foreign_keys,
                    "indexes": indexes
                }
                
                for column in columns:
                    column_info = {
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column["nullable"],
                        "default": column.get("default"),
                        "description": f"列 {column['name']} 类型为 {column['type']}"
                    }
                    table_info["columns"].append(column_info)
                
                schema_info["tables"][table_name] = table_info
            
            self.schema_info = schema_info
            
            # Save to file for future use
            os.makedirs(os.path.dirname(config.SQL_TABLES_INFO_PATH), exist_ok=True)
            with open(config.SQL_TABLES_INFO_PATH, 'w', encoding='utf-8') as f:
                json.dump(schema_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Inspected database schema: {len(tables)} tables found")
            
        except Exception as e:
            logger.error(f"Error inspecting database schema: {e}")
    
    def _create_sample_schema_info(self):
        """创建示例结构信息用于演示"""
        sample_schema = {
            "tables": {
                "customers": {
                    "description": "客户信息表",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "nullable": False, "description": "客户ID"},
                        {"name": "name", "type": "VARCHAR(100)", "nullable": False, "description": "客户姓名"},
                        {"name": "email", "type": "VARCHAR(255)", "nullable": True, "description": "邮箱地址"},
                        {"name": "phone", "type": "VARCHAR(20)", "nullable": True, "description": "电话号码"},
                        {"name": "registration_date", "type": "TIMESTAMP", "nullable": False, "description": "注册日期"},
                        {"name": "status", "type": "VARCHAR(20)", "nullable": False, "description": "客户状态(active/inactive)"}
                    ]
                },
                "orders": {
                    "description": "订单信息表",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "nullable": False, "description": "订单ID"},
                        {"name": "customer_id", "type": "INTEGER", "nullable": False, "description": "客户ID"},
                        {"name": "order_number", "type": "VARCHAR(50)", "nullable": False, "description": "订单号"},
                        {"name": "status", "type": "VARCHAR(20)", "nullable": False, "description": "订单状态"},
                        {"name": "total_amount", "type": "DECIMAL(10,2)", "nullable": False, "description": "订单总金额"},
                        {"name": "order_date", "type": "TIMESTAMP", "nullable": False, "description": "下单日期"},
                        {"name": "shipping_address", "type": "TEXT", "nullable": True, "description": "配送地址"}
                    ]
                },
                "products": {
                    "description": "产品信息表",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "nullable": False, "description": "产品ID"},
                        {"name": "name", "type": "VARCHAR(200)", "nullable": False, "description": "产品名称"},
                        {"name": "category", "type": "VARCHAR(50)", "nullable": False, "description": "产品分类"},
                        {"name": "price", "type": "DECIMAL(10,2)", "nullable": False, "description": "产品价格"},
                        {"name": "stock", "type": "INTEGER", "nullable": False, "description": "库存数量"},
                        {"name": "description", "type": "TEXT", "nullable": True, "description": "产品描述"}
                    ]
                },
                "order_items": {
                    "description": "订单项目表",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "nullable": False, "description": "订单项ID"},
                        {"name": "order_id", "type": "INTEGER", "nullable": False, "description": "订单ID"},
                        {"name": "product_id", "type": "INTEGER", "nullable": False, "description": "产品ID"},
                        {"name": "quantity", "type": "INTEGER", "nullable": False, "description": "购买数量"},
                        {"name": "unit_price", "type": "DECIMAL(10,2)", "nullable": False, "description": "单价"}
                    ]
                }
            }
        }
        
        self.schema_info = sample_schema
        
        # Save sample schema
        try:
            os.makedirs(os.path.dirname(config.SQL_TABLES_INFO_PATH), exist_ok=True)
            with open(config.SQL_TABLES_INFO_PATH, 'w', encoding='utf-8') as f:
                json.dump(sample_schema, f, ensure_ascii=False, indent=2)
            logger.info("Created sample schema info")
        except Exception as e:
            logger.error(f"Error saving sample schema: {e}")
    
    def _build_schema_prompt(self) -> str:
        """构建用于提示的结构信息"""
        if not self.schema_info or "tables" not in self.schema_info:
            return "No schema information available."
        
        schema_text = "数据库架构信息:\n\n"
        
        for table_name, table_info in self.schema_info["tables"].items():
            schema_text += f"表名: {table_name}\n"
            schema_text += f"描述: {table_info.get('description', '')}\n"
            schema_text += "列信息:\n"
            
            for column in table_info.get("columns", []):
                schema_text += f"  - {column['name']} ({column['type']}): {column.get('description', '')}\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def generate_sql(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """根据自然语言问题生成SQL查询"""
        try:
            if not self.enabled:
                return None, "Text2SQL功能未启用"
            
            if not self.local_pipe:
                return self._fallback_sql_generation(question)
            
            schema_prompt = self._build_schema_prompt()
            
            # Create prompt template
            prompt_template = PromptTemplate(
                input_variables=["schema", "question"],
                template="""你是一个SQL查询生成专家。根据提供的数据库架构和用户问题，生成准确的SQL查询。

{schema}

用户问题: {question}

请根据以上信息生成SQL查询，要求：
1. 只返回SQL查询语句，不要包含其他解释
2. 使用标准PostgreSQL语法
3. 确保查询语法正确
4. 如果问题不明确或无法生成SQL，返回"无法生成SQL查询"

SQL查询:"""
            )
            
            # Generate prompt
            prompt = prompt_template.format(
                schema=schema_prompt,
                question=question
            )
            
            # Get response from LLM
            result = self.local_pipe(prompt, max_new_tokens=256, temperature=0.1)
            sql_text = result[0]["generated_text"][len(prompt):].strip()
            sql_query = self.sql_parser.parse(sql_text)
            
            # Validate SQL
            if self._validate_sql(sql_query):
                return sql_query, None
            else:
                return None, "生成的SQL查询无效"
        
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None, f"SQL生成失败: {str(e)}"
    
    def _fallback_sql_generation(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """无LLM时的兜底SQL生成"""
        # Simple pattern matching for common queries
        question_lower = question.lower()
        
        # Common query patterns
        if "订单" in question and "状态" in question:
            return "SELECT id, order_number, status, total_amount FROM orders WHERE status = 'pending';", None
        elif "客户" in question and "信息" in question:
            return "SELECT id, name, email, phone FROM customers LIMIT 10;", None
        elif "产品" in question and ("库存" in question or "数量" in question):
            return "SELECT id, name, stock FROM products WHERE stock > 0;", None
        else:
            return None, "无法识别查询意图，请使用更具体的问题"
    
    def _validate_sql(self, sql_query: str) -> bool:
        """校验SQL查询语法"""
        try:
            if not sql_query or sql_query.strip() == "":
                return False
            
            if "无法生成" in sql_query:
                return False
            
            # Basic SQL validation
            sql_lower = sql_query.lower().strip()
            
            # Check for dangerous operations
            dangerous_keywords = ['drop', 'delete', 'truncate', 'insert', 'update', 'alter', 'create']
            if any(keyword in sql_lower for keyword in dangerous_keywords):
                logger.warning(f"Potentially dangerous SQL query blocked: {sql_query}")
                return False
            
            # Must start with SELECT
            if not sql_lower.startswith('select'):
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating SQL: {e}")
            return False
    
    def execute_sql(self, sql_query: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """执行SQL查询并返回结果"""
        try:
            if not self.enabled or not self.engine:
                return None, "数据库连接未配置"
            
            if not self._validate_sql(sql_query):
                return None, "SQL查询无效或包含危险操作"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                
                # Convert to list of dictionaries
                columns = result.keys()
                rows = result.fetchall()
                
                data = []
                for row in rows:
                    row_dict = {}
                    for i, column in enumerate(columns):
                        value = row[i]
                        # Handle different data types
                        if hasattr(value, 'isoformat'):  # datetime objects
                            value = value.isoformat()
                        elif isinstance(value, (bytes, memoryview)):
                            value = str(value)
                        row_dict[column] = value
                    data.append(row_dict)
                
                logger.info(f"SQL query executed successfully, returned {len(data)} rows")
                return data, None
        
        except Exception as e:
            error_msg = f"SQL执行失败: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """处理自然语言问题并返回SQL结果"""
        try:
            if not self.enabled:
                return {
                    "success": False,
                    "error": "Text2SQL功能已禁用",
                    "sql_query": None,
                    "data": None
                }
            
            # Generate SQL
            sql_query, sql_error = self.generate_sql(question)
            
            if sql_error or not sql_query:
                return {
                    "success": False,
                    "error": sql_error or "无法生成SQL查询",
                    "sql_query": sql_query,
                    "data": None
                }
            
            # Execute SQL
            data, exec_error = self.execute_sql(sql_query)
            
            if exec_error:
                return {
                    "success": False,
                    "error": exec_error,
                    "sql_query": sql_query,
                    "data": None
                }
            
            return {
                "success": True,
                "error": None,
                "sql_query": sql_query,
                "data": data,
                "row_count": len(data) if data else 0
            }
        
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "success": False,
                "error": f"处理问题时发生错误: {str(e)}",
                "sql_query": None,
                "data": None
            }
    
    def get_sample_questions(self) -> List[str]:
        """获取用于测试的示例问题"""
        return [
            "查询所有客户的信息",
            "显示待处理的订单",
            "查看库存不足的产品",
            "统计每个客户的订单数量",
            "查询最近一周的订单",
            "显示销量最好的产品",
            "查询特定客户的订单历史",
            "统计各个产品类别的销售额"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """获取Text2SQL服务状态"""
        return {
            "enabled": self.enabled,
            "database_connected": self.engine is not None,
            "llm_available": self.local_pipe is not None,
            "schema_loaded": bool(self.schema_info),
            "table_count": len(self.schema_info.get("tables", {}))
        }