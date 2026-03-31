"""Simple test script for Healthcare RAG API"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_specialties():
    """Test specialties endpoint"""
    print("\n=== Testing Specialties ===")
    response = requests.get(f"{BASE_URL}/api/specialties")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total specialties: {data['total']}")
    print("Top 5 specialties:")
    for spec in data['specialties'][:5]:
        print(f"  - {spec['name']}: {spec['count']} câu hỏi")

def test_stats():
    """Test stats endpoint"""
    print("\n=== Testing Stats ===")
    response = requests.get(f"{BASE_URL}/api/stats")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_chat(query: str):
    """Test chat endpoint"""
    print(f"\n=== Testing Chat: '{query}' ===")
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "query": query,
            "include_sources": True
        }
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    
    print(f"\nChuyên khoa: {data['specialty']}")
    print(f"Độ tin cậy: {data['confidence']:.2f}")
    print(f"\nCâu trả lời:\n{data['answer'][:300]}...")
    
    if data.get('sources'):
        print(f"\nNguồn tham khảo ({len(data['sources'])} nguồn):")
        for i, source in enumerate(data['sources'][:3], 1):
            print(f"\n  {i}. Type: {source['type']} | Score: {source['score']:.2f}")
            if source['type'] == 'qa':
                print(f"     Question: {source['question'][:80]}...")
            else:
                print(f"     Title: {source['title'][:80]}...")

if __name__ == "__main__":
    print("=" * 60)
    print("Healthcare RAG API - Test Suite")
    print("=" * 60)
    
    try:
        # Test basic endpoints
        test_health()
        test_stats()
        test_specialties()
        
        # Test chat with various queries
        test_queries = [
            "Bé 2 tuổi sốt 38.5 độ, tôi phải làm gì?",
            "Đau đầu thường xuyên, buồn nôn",
            "Làm sao để giảm cân hiệu quả?"
        ]
        
        for query in test_queries:
            test_chat(query)
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server.")
        print("Make sure the API is running at http://localhost:8000")
        print("Run: uvicorn api.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
