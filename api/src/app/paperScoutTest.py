#!/usr/bin/env python3
"""
PaperScout Test Script
Run this to test your PaperScout system with sample papers
"""

import asyncio
import httpx
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

async def test_paperscout():
    """Test the full PaperScout pipeline"""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("🔬 Testing PaperScout Pipeline\n")
        
        # 1. Health check
        print("1. Checking API health...")
        try:
            response = await client.get(f"{API_BASE}/health")
            if response.status_code == 200:
                print("✅ API is healthy")
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return
        
        # 2. Load test papers
        print("\n2. Loading test papers...")
        test_data_path = Path("test_papers.json")
        if not test_data_path.exists():
            print("❌ test_papers.json not found. Create it first!")
            return
        
        with open(test_data_path) as f:
            papers_data = json.load(f)
        
        print(f"📄 Loaded {len(papers_data['papers'])} test papers")
        
        # 3. Create papers in batch
        print("\n3. Creating papers in database...")
        response = await client.post(
            f"{API_BASE}/api/papers/batch",
            json=papers_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Created {result['created_count']} papers")
            if result['failed_count'] > 0:
                print(f"⚠️  {result['failed_count']} papers failed to create")
            paper_ids = result['created_paper_ids']
        else:
            print(f"❌ Failed to create papers: {response.status_code}")
            print(response.text)
            return
        
        # 4. Start processing first paper
        if paper_ids:
            test_paper_id = paper_ids[0]
            print(f"\n4. Processing paper {test_paper_id}...")
            
            response = await client.post(
                f"{API_BASE}/api/processing/papers/{test_paper_id}/full"
            )
            
            if response.status_code == 200:
                print("✅ Processing started")
            else:
                print(f"❌ Failed to start processing: {response.status_code}")
                return
            
            # 5. Monitor processing status
            print("\n5. Monitoring processing status...")
            max_wait = 300  # 5 minutes max
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                response = await client.get(
                    f"{API_BASE}/api/processing/status/{test_paper_id}"
                )
                
                if response.status_code == 200:
                    status = response.json()
                    print(f"📊 Status: {status['status']} - {status['message']}")
                    
                    if status['status'] == 'completed':
                        print("✅ Processing completed!")
                        break
                    elif status['status'] == 'failed':
                        print("❌ Processing failed!")
                        break
                else:
                    print(f"❌ Failed to get status: {response.status_code}")
                
                await asyncio.sleep(10)  # Wait 10 seconds
            else:
                print("⏰ Processing timeout - check logs")
        
        # 6. Test search functionality
        print("\n6. Testing search...")
        
        # Search for attention-related papers
        response = await client.get(
            f"{API_BASE}/api/search/",
            params={"q": "attention", "limit": 5}
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"🔍 Found {len(results['papers'])} papers matching 'attention'")
            for paper in results['papers']:
                print(f"   - {paper['title']}")
        else:
            print(f"❌ Search failed: {response.status_code}")
        
        # 7. Get processing stats
        print("\n7. Getting processing statistics...")
        response = await client.get(f"{API_BASE}/api/processing/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"📈 Processing Stats:")
            print(f"   Total papers: {stats['total_papers']}")
            print(f"   Completed: {stats['completed']}")
            print(f"   Processing: {stats['processing']}")
            print(f"   Pending: {stats['pending']}")
            print(f"   Failed: {stats['failed']}")
        
        # 8. List papers with content creation filter
        print("\n8. Finding papers ready for content creation...")
        response = await client.get(
            f"{API_BASE}/api/papers/unused/content",
            params={"min_relevance": 5, "limit": 3}
        )
        
        if response.status_code == 200:
            papers = response.json()
            print(f"📝 Found {len(papers)} papers ready for content:")
            for paper in papers:
                relevance = paper.get('business_relevance_score', 'N/A')
                topics = ', '.join(paper.get('main_topics', [])[:3])
                print(f"   - {paper['title']} (Relevance: {relevance}, Topics: {topics})")
        
        print("\n🎉 Test completed!")

def create_test_data():
    """Create the test_papers.json file"""
    test_data = {
        "papers": [
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/pdf/1706.03762.pdf",
                "arxiv_id": "1706.03762"
            },
            {
                "title": "YOLO: Real-Time Object Detection", 
                "url": "https://arxiv.org/pdf/1506.02640.pdf",
                "arxiv_id": "1506.02640"
            },
            {
                "title": "Deep Reinforcement Learning with Double Q-learning",
                "url": "https://arxiv.org/pdf/1509.06461.pdf", 
                "arxiv_id": "1509.06461"
            }
        ]
    }
    
    with open("test_papers.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print("✅ Created test_papers.json")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-data":
        create_test_data()
    else:
        print("Make sure your PaperScout services are running:")
        print("cd paperscout && docker-compose up")
        print("\nThen run: python test_paperscout.py")
        print("Or to create test data: python test_paperscout.py create-data")
        print()
        
        asyncio.run(test_paperscout())
