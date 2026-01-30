"""
hospital_faqs Q-only 임베딩 마이그레이션 스크립트

기존 hospital_faqs의 Q+A 전체 임베딩을 Q(질문)만 임베딩으로 변환하고,
카테고리 메타데이터를 추가합니다.

사용법:
  python migrate_qa_embeddings.py --dry-run    # 변경 없이 미리보기
  python migrate_qa_embeddings.py              # 실제 마이그레이션 실행
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from database.supabase_client import SupabaseManager
from utils.embeddings import get_embedding
from config.settings import HOSPITAL_FAQS_TABLE

# ──────────────────────────────────────────────
# 카테고리 분류 키워드
# ──────────────────────────────────────────────
CANCER_KEYWORDS = [
    "암", "항암", "온열", "고주파", "면역", "종양", "전이", "요양",
    "보조치료", "암세포", "방사선", "화학요법", "NK세포", "미슬토",
    "셀레늄", "고용량비타민", "비타민C", "온코써미아", "하이퍼써미아",
]
NERVE_KEYWORDS = [
    "자율신경", "신경", "두통", "어지럼", "불면", "스트레스",
    "이명", "손발저림", "실조증", "교감신경", "부교감신경",
]


def classify_category(text: str) -> str:
    """질문 텍스트로 카테고리를 분류합니다."""
    normalized = text.replace(" ", "")
    if any(kw in normalized for kw in CANCER_KEYWORDS):
        return "cancer"
    if any(kw in normalized for kw in NERVE_KEYWORDS):
        return "nerve"
    return "general"


def parse_question(content: str) -> str:
    """'Q: ...\\nA: ...' 형식에서 질문만 추출합니다."""
    if "Q:" in content and "A:" in content:
        q_start = content.index("Q:") + 2
        a_start = content.index("A:")
        return content[q_start:a_start].strip()
    # Q: 형식이 아닌 경우 첫 줄 사용
    return content.split("\n")[0].strip()


def backup_data(rows, backup_path):
    """기존 데이터를 JSON 파일로 백업합니다."""
    backup = []
    for row in rows:
        item = {
            "id": row["id"],
            "content": row["content"],
            "metadata": row.get("metadata"),
        }
        backup.append(item)

    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup, f, ensure_ascii=False, indent=2)
    print(f"백업 완료: {backup_path} ({len(backup)}건)")


def main():
    parser = argparse.ArgumentParser(description="hospital_faqs Q-only 임베딩 마이그레이션")
    parser.add_argument("--dry-run", action="store_true", help="변경 없이 미리보기만 실행")
    args = parser.parse_args()

    db = SupabaseManager()

    # 1. 전체 데이터 조회
    print("hospital_faqs 데이터 조회 중...")
    response = db.client.table(HOSPITAL_FAQS_TABLE).select("id, content, metadata").execute()
    rows = response.data
    print(f"총 {len(rows)}건 조회됨")

    if not rows:
        print("데이터가 없습니다. 종료합니다.")
        return

    # 2. 백업
    backup_path = os.path.join(os.path.dirname(__file__), "hospital_faqs_backup.json")
    if not args.dry_run:
        backup_data(rows, backup_path)

    # 3. 파싱 및 분류 미리보기
    parsed = []
    for row in rows:
        content = row["content"]
        question = parse_question(content)
        category = classify_category(question)
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        parsed.append({
            "id": row["id"],
            "content": content,
            "question": question,
            "category": category,
            "metadata": metadata,
        })

    # 통계
    cat_counts = {}
    for p in parsed:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1

    print(f"\n카테고리 분포: {cat_counts}")
    print(f"\n처음 5건 미리보기:")
    for p in parsed[:5]:
        q_preview = p["question"][:60]
        print(f"  [{p['category']}] {q_preview}")

    if args.dry_run:
        print(f"\n[DRY-RUN] 전체 {len(parsed)}건 파싱 결과:")
        for i, p in enumerate(parsed):
            print(f"  {i+1}. [{p['category']}] Q: {p['question'][:80]}")
        print("\n--dry-run 모드이므로 DB 변경 없이 종료합니다.")
        return

    # 4. 실제 마이그레이션
    print(f"\n{'='*60}")
    print(f"Q-only 임베딩 마이그레이션 시작 ({len(parsed)}건)")
    print(f"{'='*60}")

    success = 0
    errors = 0

    for i, p in enumerate(parsed):
        try:
            # Q 텍스트로 새 임베딩 생성
            new_embedding = get_embedding(p["question"])
            if not new_embedding:
                print(f"  [{i+1}/{len(parsed)}] SKIP - 임베딩 생성 실패: {p['question'][:40]}")
                errors += 1
                continue

            # metadata에 category 추가
            updated_metadata = dict(p["metadata"])
            updated_metadata["category"] = p["category"]

            # DB 업데이트
            db.update_row(HOSPITAL_FAQS_TABLE, p["id"], {
                "embedding": new_embedding,
                "metadata": updated_metadata,
            })

            success += 1
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(parsed)}] OK - [{p['category']}] {p['question'][:50]}")

            # API rate limit 대응
            if (i + 1) % 20 == 0:
                time.sleep(1)

        except Exception as e:
            print(f"  [{i+1}/{len(parsed)}] ERROR - {p['question'][:40]}: {e}")
            errors += 1
            time.sleep(2)

    print(f"\n{'='*60}")
    print(f"마이그레이션 완료: 성공 {success}건, 실패 {errors}건")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
