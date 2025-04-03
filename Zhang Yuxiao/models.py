import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 知识库目录配置
KNOWLEDGE_BASE = {
    "CancerGov": "1_CancerGov_QA",
    "GARD": "2_GARD_QA",
    "GHR": "3_GHR_QA",
    "MPlus_Health": "4_MPlus_Health_Topics_QA",
    "NIDDK": "5_NIDDK_QA",
    "NINDS": "6_NINDS_QA",
    "SeniorHealth": "7_SeniorHealth_QA",
    "NHLBI": "8_NHLBI_QA_XML",
    "CDC": "9_CDC_QA",
    "MPlus_ADAM": "10_MPlus_ADAM_QA",
    "MPlusDrugs": "11_MPlusDrugs_QA",
    "MPlusHerbs": "12_MPlusHerbsSupplements_QA"
}

def parse_knowledge_base(base_path):
    """解析整个知识库目录"""
    qa_pairs = []
    total_files = 0
    
    # 遍历所有知识库模块
    for module_name, folder_name in KNOWLEDGE_BASE.items():
        module_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(module_path):
            print(f"警告：跳过不存在的模块目录 {module_name} -> {module_path}")
            continue
        
        # 遍历模块目录下的XML文件
        for filename in os.listdir(module_path):
            if not filename.lower().endswith('.xml'):
                continue
                
            file_path = os.path.join(module_path, filename)
            try:
                pairs = parse_single_xml(file_path)
                qa_pairs.extend(pairs)
                total_files += 1
                print(f"成功加载 {filename} ({len(pairs)} 条问答)")
            except Exception as e:
                print(f"解析失败 {filename}: {str(e)}")
                continue
    
    if not qa_pairs:
        raise ValueError(f"知识库目录中未找到有效数据，请检查：{base_path}")
    
    print(f"\n知识库加载完成！共加载 {len(qa_pairs)} 条问答，来自 {total_files} 个文件")
    return qa_pairs

def parse_single_xml(xml_path):
    """解析单个XML文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    qa_pairs = []
    qa_pairs_container = root.find('QAPairs')
    
    if qa_pairs_container is None:
        return []
    
    for qa_pair in qa_pairs_container.findall('QAPair'):
        try:
            # 提取问题
            question_elem = qa_pair.find('Question')
            if question_elem is None:
                continue
                
            question = question_elem.text.strip() if question_elem.text else ""
            qid = question_elem.get('qid', '')
            
            # 提取答案
            answer_elem = qa_pair.find('Answer')
            if answer_elem is not None and answer_elem.text is not None:
                answer = re.sub(r'\s+', ' ', answer_elem.text.strip())
            else:
                answer = ""
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "qid": qid,
                "source": os.path.basename(xml_path)  # 记录来源文件
            })
            
        except Exception as e:
            print(f"解析异常已跳过：{str(e)}")
            continue
    
    return qa_pairs

class MedicalRetriever:
    def __init__(self, base_path, use_bm25=True):
        # 加载知识库
        self.qa_pairs = parse_knowledge_base(base_path)
        self.questions = [p["question"] for p in self.qa_pairs]
        
        # 文本处理管道
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            stop_words=None,
            ngram_range=(1, 3),
            min_df=2,  # 过滤低频词
            max_df=0.85,
            max_features=20000,
            token_pattern=r"(?u)\b\w[\w'-]+\b"
        )
        
        # 构建文档矩阵
        processed_questions = [self._preprocess(q) for q in self.questions]
        self.doc_vectors = self.vectorizer.fit_transform(processed_questions)
        
        # 验证特征工程
        if len(self.vectorizer.get_feature_names_out()) == 0:
            raise ValueError("特征提取失败，请检查预处理逻辑")
        
        print(f"\n特征工程完成，提取特征词数量：{len(self.vectorizer.vocabulary_)}")
        print("示例特征词：", list(self.vectorizer.vocabulary_.keys())[::500])
        
        # BM25初始化
        self.use_bm25 = use_bm25
        if use_bm25:
            self._init_bm25_params()

    def _preprocess(self, text):
        """优化的预处理"""
        text = re.sub(r"[^\w\s'-]", '', text.lower())
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text):
        """医学专业分词"""
        return re.findall(r"\b[\w']+(?:-[\w']+)+\b|\b[\w']{2,}\b", text)

    def _init_bm25_params(self):
        """BM25参数计算"""
        tf = self.doc_vectors.toarray()
        df = np.sum(tf > 0, axis=0)
        self.idf = np.log((len(self.questions) - df + 0.5) / (df + 0.5))
        self.avg_dl = np.mean(np.sum(tf, axis=1))

    def _bm25_score(self, query_vec, doc_vec):
        """BM25评分计算"""
        k1, b = 1.5, 0.75
        tf = doc_vec.toarray().flatten()
        q_tf = query_vec.toarray().flatten()
        
        doc_len = np.sum(tf)
        norm_factor = 1 - b + b * doc_len / self.avg_dl
        
        scores = self.idf * ((tf * (k1 + 1)) / (tf + k1 * norm_factor)) * q_tf
        return np.sum(scores)

    def retrieve_topk(self, query, k=5):
        """混合检索方法"""
        processed_query = self._preprocess(query)
        query_vec = self.vectorizer.transform([processed_query])
        
        if self.use_bm25:
            scores = [self._bm25_score(query_vec, doc) for doc in self.doc_vectors]
        else:
            scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        
        topk_indices = np.argsort(scores)[-k:][::-1]
        
        return [{
            "question": self.qa_pairs[i]["question"],
            "answer": self.qa_pairs[i]["answer"],
            "qid": self.qa_pairs[i]["qid"],
            "score": float(scores[i]),
            "source": self.qa_pairs[i]["source"]
        } for i in topk_indices]

# 使用示例
if __name__ == "__main__":
    try:
        # 初始化检索器（指向MedQuAD-master目录）
        retriever = MedicalRetriever(
            base_path="./MedQuAD-master",
            use_bm25=True
        )
        
        # 示例查询
        #query = getQuery()
        query = "What are the polysystic ovary syndrome?"
        results = retriever.retrieve_topk(query, k=3)
        
        # 增强输出
        print("\n" + "="*60 + " 检索结果 " + "="*60)
        print(f"查询: {query}")
        for i, res in enumerate(results):
            print(f"\nTOP {i+1} (相似度: {res['score']:.4f})")
            print(f"[来源文件] {res['source']}")
            print(f"[问题ID] {res['qid']}")
            print(f"[问题] {res['question']}")
            print(f"[答案摘要] {res['answer'][:350]}...")
        print("="*130)
        
    except Exception as e:
        print("\n" + "!"*30 + " 系统错误 " + "!"*30)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        print("\n排查建议:")
        print("1. 确认MedQuAD-master目录结构符合要求")
        print("2. 检查至少有一个子目录包含有效XML文件")
        print("3. 尝试设置use_bm25=False使用TF-IDF算法")
        print("4. 查看控制台输出的特征词示例是否包含医学术语")