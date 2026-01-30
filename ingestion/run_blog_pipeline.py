"""
Naver Blog to Vector DB Pipeline

Phase 1: Crawl baravo blog posts
Phase 2: Save to documents table
Phase 3: Transform to Q/A using Gemini
Phase 4: Save to hospital_faqs table

Usage:
    python ingestion/run_blog_pipeline.py
    
    # Dry run (crawl only, don't save)
    python ingestion/run_blog_pipeline.py --dry-run
    
    # Limit pages
    python ingestion/run_blog_pipeline.py --max-pages 5
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import List, Dict
from ingestion.naver_blog_crawler import NaverBlogCrawler
from ingestion.qa_transformer import QATransformer, run_qa_transformation
from database.supabase_client import SupabaseManager


def save_to_documents(posts: List[Dict], db: SupabaseManager, dry_run: bool = False) -> bool:
    """크롤링된 글을 documents 테이블에 저장"""
    if not posts:
        print("[Pipeline] No posts to save")
        return False
        
    print(f"\n{'='*60}")
    print("[Phase 2] Saving to documents table")
    print(f"{'='*60}")
    
    # Format for documents table
    documents = []
    for post in posts:
        doc = {
            "content": post['content'],
            "metadata": {
                "source": post['url'],
                "title": post['title'],
                "type": "blog",
                "author": post.get('author', 'baravo'),
                "published_date": post.get('date', ''),
                "thumbnail": post.get('thumbnail', ''),
                "post_id": post.get('post_id', ''),
            }
        }
        documents.append(doc)
    
    if dry_run:
        print(f"[DRY RUN] Would save {len(documents)} documents")
        for doc in documents[:2]:
            print(f"  - {doc['metadata']['title'][:50]}...")
        return True
        
    try:
        db.insert_documents(documents, table_name="documents")
        print(f"[Pipeline] [OK] Saved {len(documents)} documents")
        return True
    except Exception as e:
        print(f"[Pipeline] [ERROR] Error saving documents: {e}")
        return False


def transform_and_save_to_faqs(posts: List[Dict], db: SupabaseManager, dry_run: bool = False) -> bool:
    """글을 Q/A로 변환하여 hospital_faqs 테이블에 저장"""
    if not posts:
        print("[Pipeline] No posts to transform")
        return False
        
    print(f"\n{'='*60}")
    print("[Phase 3] Transforming to Q/A format")
    print(f"{'='*60}")
    
    transformer = QATransformer()
    
    # Transform to Q/A
    qa_list = transformer.transform_batch(posts, delay=1.0)
    
    if not qa_list:
        print("[Pipeline] No Q/A generated")
        return False
        
    print(f"\n{'='*60}")
    print("[Phase 4] Saving to hospital_faqs table")
    print(f"{'='*60}")
    
    # Format for hospital_faqs table
    faqs = transformer.format_for_faqs_table(qa_list)
    
    if dry_run:
        print(f"[DRY RUN] Would save {len(faqs)} FAQ entries")
        for faq in faqs[:2]:
            print(f"\n  Q: {faq['metadata']['question']}")
            print(f"  A: {faq['metadata']['answer'][:80]}...")
        return True
        
    try:
        db.insert_documents(faqs, table_name="hospital_faqs")
        print(f"[Pipeline] ✓ Saved {len(faqs)} FAQs")
        return True
    except Exception as e:
        print(f"[Pipeline] ✗ Error saving FAQs: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Naver Blog to Vector DB Pipeline')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Crawl only, do not save to database')
    parser.add_argument('--max-pages', type=int, default=10,
                        help='Maximum pages to crawl (default: 10)')
    parser.add_argument('--skip-documents', action='store_true',
                        help='Skip saving to documents table')
    parser.add_argument('--skip-faqs', action='store_true',
                        help='Skip transforming and saving to hospital_faqs')
    parser.add_argument('--blog-id', type=str, default='baravo',
                        help='Naver blog ID (default: baravo)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Naver Blog to Vector DB Pipeline")
    print(f"{'='*60}")
    print(f"Blog ID: {args.blog_id}")
    print(f"Max pages: {args.max_pages}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}\n")
    
    # Phase 1: Crawl
    print("[Phase 1] Crawling blog posts")
    print("-" * 60)
    
    crawler = NaverBlogCrawler(blog_id=args.blog_id)
    posts = crawler.crawl_all_posts(max_posts=args.max_pages * 10)  # max_pages * 10 posts per page roughly
    
    if not posts:
        print("\n[Pipeline] No posts crawled. Exiting.")
        return
        
    print(f"\n[Pipeline] Successfully crawled {len(posts)} posts")
    
    # Initialize DB manager if needed
    db = None if args.dry_run else SupabaseManager()
    
    # Phase 2: Save raw documents
    if not args.skip_documents:
        save_to_documents(posts, db, args.dry_run)
    else:
        print("\n[Pipeline] Skipping documents table (as requested)")
    
    # Phase 3 & 4: Transform and save Q/A
    if not args.skip_faqs:
        transform_and_save_to_faqs(posts, db, args.dry_run)
    else:
        print("\n[Pipeline] Skipping hospital_faqs table (as requested)")
    
    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
