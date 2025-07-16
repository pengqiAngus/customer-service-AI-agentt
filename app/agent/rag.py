import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PDFPlumberLoader,
    CSVLoader,
    JSONLoader
)
from langchain.schema import Document

from app.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """RAG Knowledge Base for customer service documents"""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and vector store"""
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            
            # Initialize ChromaDB
            os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
            client = chromadb.PersistentClient(path=config.VECTOR_STORE_PATH)
            
            # Create or get collection
            try:
                self.vector_store = client.get_collection("knowledge_base")
                logger.info("Loaded existing vector store")
            except:
                self.vector_store = client.create_collection(
                    name="knowledge_base",
                    metadata={"description": "Customer service knowledge base"}
                )
                logger.info("Created new vector store")
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}")
            raise
    
    def _create_sample_knowledge_files(self):
        """Create sample knowledge files if none exist"""
        try:
            knowledge_path = Path(config.KNOWLEDGE_BASE_PATH)
            knowledge_path.mkdir(exist_ok=True)
            
            # Sample FAQ file
            faq_content = """
# 客户服务常见问题

## 退款相关
Q: 如何申请退款？
A: 您可以通过以下步骤申请退款：
1. 登录您的账户
2. 找到对应订单
3. 点击"申请退款"按钮
4. 填写退款原因
5. 提交申请
退款通常在3-5个工作日内处理完成。

Q: 退款多久能到账？
A: 退款到账时间取决于您的支付方式：
- 支付宝/微信：1-3个工作日
- 银行卡：3-7个工作日
- 信用卡：7-15个工作日

## 订单相关
Q: 如何查询订单状态？
A: 您可以通过以下方式查询订单：
1. 登录官网账户中心
2. 使用订单号在首页查询
3. 联系在线客服查询
4. 拨打客服热线：400-XXX-XXXX

Q: 订单可以修改吗？
A: 订单修改需要在发货前：
- 未发货：可以修改收货地址、联系方式
- 已发货：无法修改，建议拒收后重新下单
- 特殊情况：请联系客服协助处理

## 配送相关
Q: 配送范围包括哪些地区？
A: 我们支持全国配送，包括：
- 一线城市：当日达/次日达
- 二三线城市：2-3天送达
- 偏远地区：3-7天送达
- 港澳台地区：7-15天送达

Q: 配送费用如何计算？
A: 配送费用标准：
- 订单金额≥99元：免运费
- 订单金额<99元：根据重量和距离计算
- 加急配送：额外收取加急费用
- 偏远地区：可能产生额外费用

## 产品相关
Q: 产品质量保证？
A: 我们提供全面的质量保证：
- 7天无理由退货
- 15天换货保证
- 1年质量保修
- 假一赔三承诺

Q: 如何辨别产品真伪？
A: 正品验证方法：
1. 查看防伪标签
2. 扫描二维码验证
3. 官网序列号查询
4. 客服人工验证
"""
            
            faq_file = knowledge_path / "faq.md"
            if not faq_file.exists():
                with open(faq_file, 'w', encoding='utf-8') as f:
                    f.write(faq_content)
                logger.info("Created sample FAQ file")
            
            # Sample policy file
            policy_content = {
                "company_policies": {
                    "return_policy": {
                        "title": "退货政策",
                        "description": "客户可在收到商品后7天内申请无理由退货",
                        "conditions": [
                            "商品未使用且包装完整",
                            "不影响二次销售",
                            "非定制化商品",
                            "在有效期内申请"
                        ],
                        "process": [
                            "联系客服申请",
                            "填写退货单",
                            "寄回商品",
                            "审核通过后退款"
                        ]
                    },
                    "shipping_policy": {
                        "title": "配送政策",
                        "description": "全国范围配送服务",
                        "delivery_time": {
                            "同城": "24小时内",
                            "省内": "2-3天",
                            "跨省": "3-5天",
                            "偏远地区": "5-7天"
                        },
                        "shipping_fee": {
                            "免运费门槛": "99元",
                            "标准运费": "8-15元",
                            "加急费": "10-20元"
                        }
                    },
                    "warranty_policy": {
                        "title": "质保政策",
                        "description": "提供全面的产品质量保证",
                        "warranty_period": "1年",
                        "coverage": [
                            "制造缺陷",
                            "材料问题",
                            "功能故障"
                        ],
                        "exclusions": [
                            "人为损坏",
                            "自然磨损",
                            "超出保修期"
                        ]
                    }
                }
            }
            
            policy_file = knowledge_path / "policies.json"
            if not policy_file.exists():
                with open(policy_file, 'w', encoding='utf-8') as f:
                    json.dump(policy_content, f, ensure_ascii=False, indent=2)
                logger.info("Created sample policy file")
                
        except Exception as e:
            logger.error(f"Error creating sample knowledge files: {e}")
    
    def load_documents(self, file_path: Optional[str] = None) -> List[Document]:
        """Load documents from knowledge base"""
        documents = []
        
        try:
            if file_path:
                # Load specific file
                documents.extend(self._load_single_file(file_path))
            else:
                # Load all files from knowledge base directory
                knowledge_path = Path(config.KNOWLEDGE_BASE_PATH)
                if not knowledge_path.exists():
                    self._create_sample_knowledge_files()
                
                for file_path in knowledge_path.rglob("*"):
                    if file_path.is_file():
                        documents.extend(self._load_single_file(str(file_path)))
            
            logger.info(f"Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        """Load a single file based on its extension"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.pdf':
                loader = PDFPlumberLoader(file_path)
            elif file_ext == '.csv':
                loader = CSVLoader(file_path)
            elif file_ext in ['.json', '.jsonl']:
                loader = JSONLoader(file_path, jq_schema='.')
            elif file_ext == '.md':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return []
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source_file'] = file_path
                doc.metadata['file_type'] = file_ext
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []
    
    def build_index(self, rebuild: bool = False) -> bool:
        """Build or rebuild the vector index"""
        try:
            # Check if index already exists
            if not rebuild and self.vector_store.count() > 0:
                logger.info("Vector index already exists. Use rebuild=True to rebuild.")
                return True
            
            # Clear existing index if rebuilding
            if rebuild:
                try:
                    self.vector_store.delete()
                    client = chromadb.PersistentClient(path=config.VECTOR_STORE_PATH)
                    self.vector_store = client.create_collection(
                        name="knowledge_base",
                        metadata={"description": "Customer service knowledge base"}
                    )
                except:
                    pass
            
            # Load documents
            documents = self.load_documents()
            
            if not documents:
                logger.warning("No documents found to index")
                return False
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(texts)} chunks")
            
            # Process in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Prepare batch data
                ids = [f"doc_{i + j}" for j in range(len(batch))]
                documents_text = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(documents_text).tolist()
                
                # Add to vector store
                self.vector_store.add(
                    ids=ids,
                    documents=documents_text,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            
            logger.info(f"Successfully indexed {len(texts)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            if self.vector_store.count() == 0:
                logger.warning("Vector store is empty. Building index...")
                if not self.build_index():
                    return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in vector store
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    search_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            logger.debug(f"Found {len(search_results)} relevant documents for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for a query"""
        try:
            search_results = self.search(query, top_k=5)
            
            if not search_results:
                return ""
            
            # Combine relevant documents
            context_parts = []
            current_length = 0
            
            for result in search_results:
                content = result['content']
                if current_length + len(content) <= max_context_length:
                    context_parts.append(content)
                    current_length += len(content)
                else:
                    # Add partial content if there's space
                    remaining_space = max_context_length - current_length
                    if remaining_space > 100:  # Only add if meaningful space left
                        context_parts.append(content[:remaining_space] + "...")
                    break
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add a new document to the knowledge base"""
        try:
            # Create document
            doc = Document(page_content=content, metadata=metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Get current count for ID generation
            current_count = self.vector_store.count()
            
            # Process chunks
            for i, chunk in enumerate(chunks):
                doc_id = f"doc_{current_count + i}"
                embedding = self.embedding_model.encode([chunk.page_content]).tolist()[0]
                
                self.vector_store.add(
                    ids=[doc_id],
                    documents=[chunk.page_content],
                    metadatas=[chunk.metadata],
                    embeddings=[embedding]
                )
            
            logger.info(f"Added document with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            total_documents = self.vector_store.count()
            
            # Get sample of metadata to analyze
            if total_documents > 0:
                sample_results = self.vector_store.get(limit=min(100, total_documents))
                
                # Analyze file types
                file_types = {}
                source_files = set()
                
                for metadata in sample_results['metadatas']:
                    file_type = metadata.get('file_type', 'unknown')
                    source_file = metadata.get('source_file', 'unknown')
                    
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    source_files.add(source_file)
                
                return {
                    "total_chunks": total_documents,
                    "unique_source_files": len(source_files),
                    "file_types": file_types,
                    "index_built": total_documents > 0
                }
            else:
                return {
                    "total_chunks": 0,
                    "unique_source_files": 0,
                    "file_types": {},
                    "index_built": False
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}