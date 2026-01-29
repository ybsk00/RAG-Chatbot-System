import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.generator import Generator

def test_router():
    try:
        generator = Generator()
        print("Generator initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Generator: {e}")
        return

    test_cases = [
        ("안녕하세요", "general"),
        ("진료 시간이 어떻게 되나요?", "general"),
        ("암 환자 식단 추천해줘", "cancer"),
        ("항암 치료 부작용이 뭐야?", "cancer"),
        ("자율신경 실조증 증상 알려줘", "nerve"),
        ("기립성 빈맥 증후군 치료법", "nerve"),
        ("오늘 날씨 어때?", "general"),
        ("고주파 온열 치료가 뭐야?", "cancer"),
    ]

    print("\n--- Testing Router Classification ---")
    for i, (query, expected) in enumerate(test_cases):
        category = generator._classify_query(query)
        result = "PASS" if category == expected else "FAIL"
        # Print only ASCII to avoid terminal encoding issues
        print(f"Test Case {i+1}: Expected={expected}, Predicted={category} -> [{result}]")
        if result == "FAIL":
            print(f"   Query was: {query}")

if __name__ == "__main__":
    test_router()
