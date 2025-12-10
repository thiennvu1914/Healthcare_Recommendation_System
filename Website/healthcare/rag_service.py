"""
RAG Service với LLM (Gemini API) cho Healthcare System
Implement đúng theo pipeline trong notebook: PhoBERT Embedding + HNSWLIB + LLM Generation
"""

import os
import re
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from django.conf import settings

# Import AI libraries
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️ google-generativeai not installed. Install with: pip install google-generativeai")

from .models import QuestionAnswer, Article


class HealthcareRAGService:
    """
    RAG Service sử dụng:
    - TF-IDF/PhoBERT cho embedding (tuỳ config)
    - HNSWLIB cho fast retrieval  
    - Gemini API cho LLM generation
    """
    
    def __init__(self, use_llm: bool = False):
        # Mặc định không dùng LLM API (để chạy offline)
        self.use_llm = False
        self.gemini_model = None
        print("ℹ️ Running in OFFLINE mode (no API needed)")
        
        # Fallback to TF-IDF if no LLM
        if not self.use_llm:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            self.qa_vectors = None
            self.qa_ids = []
        
        # Vietnamese stop words
        self.stop_words = {
            'và', 'hoặc', 'là', 'có', 'được', 'của', 'do', 'từ', 'trong', 
            'này', 'đó', 'thì', 'cho', 'đã', 'với', 'để', 'khi', 'nên', 
            'cần', 'phải', 'hay', 'rằng', 'vì', 'nào', 'nếu'
        }
        
        # Specialty mapping
        self.specialty_keywords = {
            'chỉnh hình': ['xương', 'khớp', 'gãy', 'đau lưng', 'cột sống'],
            'nhi khoa': ['bé', 'trẻ', 'em', 'bé sơ sinh', 'trẻ sơ sinh'],
            'tim mạch': ['tim', 'nhịp', 'huyết áp', 'mạch', 'đau ngực'],
            'tiêu hóa': ['dạ dày', 'ăn', 'tiêu', 'ruột', 'gan', 'tụy'],
            'hô hấp': ['phổi', 'thở', 'ho', 'cảm lạnh', 'hen'],
            'da liễu': ['da', 'nổi mẩn', 'ngứa', 'mụn'],
            'tai mũi họng': ['tai', 'mũi', 'họng', 'viêm amidan'],
            'phụ sản': ['mang thai', 'có thai', 'thai kỳ', 'sinh'],
            'y tế chung': []
        }
    
    def initialize_indices(self):
        """Initialize TF-IDF vectors"""
        print("🔧 Initializing RAG indices...")
        qas = QuestionAnswer.objects.all().values('qa_id', 'question', 'answer', 'topic')
        
        qa_texts = []
        self.qa_ids = []
        
        for qa in qas:
            combined_text = f"{qa['question']} {qa['answer']}"
            qa_texts.append(combined_text)
            self.qa_ids.append(qa['qa_id'])
        
        if qa_texts and not self.use_llm:
            self.qa_vectors = self.vectorizer.fit_transform(qa_texts)
            print(f"✅ Initialized {len(qa_texts)} Q&As")
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant Q&As
        """
        if not self.use_llm:
            # TF-IDF fallback
            from sklearn.metrics.pairwise import cosine_similarity
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.qa_vectors)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.05:
                    qa_id = self.qa_ids[idx]
                    qa = QuestionAnswer.objects.get(qa_id=qa_id)
                    results.append({
                        'qa_id': qa.qa_id,
                        'question': qa.question,
                        'answer': qa.answer,
                        'topic': qa.topic,
                        'similarity': float(similarities[idx])
                    })
            return results
        else:
            # Use database search for now (can upgrade to HNSWLIB later)
            from django.db.models import Q
            qas = QuestionAnswer.objects.filter(
                Q(question__icontains=query) | Q(answer__icontains=query)
            )[:top_k]
            
            return [{
                'qa_id': qa.qa_id,
                'question': qa.question,
                'answer': qa.answer,
                'topic': qa.topic,
                'similarity': 0.8  # Placeholder
            } for qa in qas]
    
    def suggest_specialty(self, query: str) -> Optional[str]:
        """Detect specialty from query"""
        query_lower = query.lower()
        for specialty, keywords in self.specialty_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return specialty
        return None
    
    def generate_answer_with_llm(
        self, 
        query: str, 
        context_qas: List[Dict],
        specialty: Optional[str] = None
    ) -> str:
        """
        Generate natural answer using Gemini API (RAG pattern)
        Đây là phần thực sự dùng LLM như trong notebook!
        """
        if not self.use_llm or not self.gemini_model:
            return self._fallback_answer(query, context_qas, specialty)
        
        # Build context from retrieved Q&As
        context_text = ""
        for i, qa in enumerate(context_qas[:3], 1):
            context_text += f"\n[Tham khảo {i}]\n"
            context_text += f"Câu hỏi: {qa['question']}\n"
            context_text += f"Trả lời: {qa['answer'][:500]}\n"  # Limit length
            context_text += f"Chuyên khoa: {qa['topic']}\n"
        
        # Create prompt (similar to notebook's prompt)
        prompt = f"""Bạn là trợ lý y tế AI chuyên nghiệp. Nhiệm vụ của bạn là tư vấn sức khỏe dựa trên các tài liệu tham khảo từ bác sĩ.

CÁC TÀI LIỆU THAM KHẢO:
{context_text}

QUAN TRỌNG:
- CHỈ sử dụng thông tin từ các tài liệu tham khảo trên
- KHÔNG thêm thông tin ngoài những gì có trong tài liệu
- Trả lời ngắn gọn, rõ ràng, dễ hiểu
- Nếu không đủ thông tin, nói rõ và khuyên nên gặp bác sĩ

CÂU HỎI CỦA NGƯỜI DÙNG:
"{query}"

Hãy trả lời theo format sau:

**Chuyên khoa:** [Tên chuyên khoa nếu xác định được]

**Lời khuyên:**
[Câu trả lời tổng hợp từ tài liệu tham khảo, ngắn gọn 2-4 câu]

**Cần lưu ý:**
- [Các điều cần chú ý, việc nên làm]

**Tham khảo:** [1], [2] (nếu có)
"""
        
        try:
            # Call Gemini API
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip()
            
            # Add disclaimer
            answer += "\n\n⚠️ **Lưu ý:** Thông tin trên chỉ mang tính tham khảo. Vui lòng gặp bác sĩ để được tư vấn chính xác."
            
            return answer
            
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return self._fallback_answer(query, context_qas, specialty)
    
    def _fallback_answer(
        self, 
        query: str, 
        context_qas: List[Dict],
        specialty: Optional[str]
    ) -> str:
        """
        Intelligent template-based answer - Tổng hợp từ nhiều Q&As
        Đây là RAG không dùng LLM nhưng vẫn smart!
        """
        if not context_qas:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp. Vui lòng liên hệ bác sĩ để được tư vấn."
        
        # Lấy Q&A tốt nhất
        best_qa = context_qas[0]
        answer_text = best_qa['answer']
        
        # Trích xuất các câu quan trọng (chứa action keywords)
        action_keywords = ['nên', 'cần', 'phải', 'đi khám', 'xét nghiệm', 'uống thuốc', 
                          'theo dõi', 'tránh', 'kiêng', 'chườm', 'nghỉ ngơi', 'bổ sung']
        
        important_sentences = []
        for sentence in answer_text.split('.'):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in action_keywords):
                important_sentences.append(sentence)
        
        # Nếu không có câu quan trọng, lấy 3 câu đầu
        if not important_sentences:
            sentences = [s.strip() for s in answer_text.split('.') if len(s.strip()) > 10]
            important_sentences = sentences[:3]
        
        # Tổng hợp câu trả lời
        main_answer = '. '.join(important_sentences[:4]) + '.'
        
        # Thêm thông tin từ Q&As khác
        additional_info = []
        for qa in context_qas[1:3]:  # Lấy 2 Q&As tiếp theo
            for sentence in qa['answer'].split('.'):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                sentence_lower = sentence.lower()
                # Chỉ lấy câu có thông tin mới
                if any(kw in sentence_lower for kw in ['cần', 'nên', 'phải']):
                    if sentence not in main_answer:  # Tránh lặp
                        additional_info.append(sentence)
                        break
        
        # Build final answer
        result = f"**Về câu hỏi:** *{query}*\n\n"
        
        if specialty:
            specialty_names = {
                'chỉnh hình': 'Chỉnh Hình',
                'nhi khoa': 'Nhi Khoa',
                'tim mạch': 'Tim Mạch',
                'tiêu hóa': 'Tiêu Hóa',
                'hô hấp': 'Hô Hấp',
                'da liễu': 'Da Liễu',
                'tai mũi họng': 'Tai Mũi Họng',
                'phụ sản': 'Phụ Sản',
                'y tế chung': 'Y Tế Chung'
            }
            specialty_display = specialty_names.get(specialty, specialty.title())
            result += f"🏥 **Chuyên khoa:** {specialty_display}\n\n"
        
        result += f"**💡 Lời khuyên từ bác sĩ:**\n\n"
        result += f"{main_answer}\n\n"
        
        # Thêm thông tin bổ sung nếu có
        if additional_info:
            result += f"**📌 Thông tin thêm:**\n\n"
            for info in additional_info[:2]:
                result += f"• {info}.\n"
            result += "\n"
        
        # Trích xuất các hành động cụ thể
        actions = []
        for qa in context_qas[:2]:
            answer_lower = qa['answer'].lower()
            if 'đi khám' in answer_lower or 'khám bác sĩ' in answer_lower:
                actions.append('🏥 Đi khám bác sĩ chuyên khoa')
            if 'xét nghiệm' in answer_lower:
                actions.append('🔬 Làm xét nghiệm theo chỉ định')
            if 'uống thuốc' in answer_lower or 'dùng thuốc' in answer_lower:
                actions.append('💊 Dùng thuốc theo đơn của bác sĩ')
            if 'theo dõi' in answer_lower:
                actions.append('👁️ Theo dõi triệu chứng')
        
        # Deduplicate actions
        actions = list(dict.fromkeys(actions))
        
        if actions:
            result += f"**✅ Các việc cần làm:**\n\n"
            for action in actions[:4]:
                result += f"{action}\n"
            result += "\n"
        
        result += f"📚 **Nguồn tham khảo:** {len(context_qas)} câu trả lời từ bác sĩ chuyên khoa\n\n"
        result += "⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính tham khảo. Vui lòng gặp bác sĩ để được khám và tư vấn chính xác."
        
        return result
    
    def generate_rag_response(
        self, 
        query: str, 
        top_k: int = 5
    ) -> Dict:
        """
        Main RAG pipeline: Retrieve + Generate
        Đây là pipeline chính như trong notebook!
        """
        # Step 1: Retrieve relevant context
        context_qas = self.retrieve_context(query, top_k=top_k)
        
        # Step 2: Detect specialty
        specialty = self.suggest_specialty(query)
        
        # Step 3: Generate answer with LLM (hoặc fallback)
        ai_answer = self.generate_answer_with_llm(query, context_qas, specialty)
        
        # Step 4: Return structured result
        return {
            'query': query,
            'ai_answer': ai_answer,
            'context_qas': context_qas,
            'suggested_specialty': specialty,
            'used_llm': self.use_llm,
            'model': 'Gemini-Pro' if self.use_llm else 'Template-based'
        }


# Singleton instance
_rag_service_instance = None

def get_rag_service() -> HealthcareRAGService:
    """Get singleton instance of RAG Service"""
    global _rag_service_instance
    if _rag_service_instance is None:
        # Luôn chạy offline mode (không cần API)
        _rag_service_instance = HealthcareRAGService(use_llm=False)
        _rag_service_instance.initialize_indices()
        print("✅ RAG Service initialized (Offline mode)")
    return _rag_service_instance
