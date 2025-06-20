#!/usr/bin/env python3
"""
Setup Test Script for LAISA
Tests connectivity to OpenAI and Pinecone services before running the Streamlit app.
"""

import os
import sys
from typing import Dict, Tuple
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()


def is_railway_environment():
    """Check if running on Railway."""
    return os.getenv("RAILWAY_ENVIRONMENT") is not None

def main():
    """Run all setup tests."""
    if is_railway_environment():
        print("Running in Railway environment - skipping interactive tests")
        # Only run critical tests that don't require user interaction
        tests = [
            ("Dependencies", run_dependency_check),
            ("Environment Variables", test_environment_variables),
        ]
    else:
        # Run all tests locally
        tests = [
            ("Dependencies", run_dependency_check),
            ("Environment Variables", test_environment_variables),
            ("OpenAI Connection", test_openai_connection),
            ("OpenAI Embeddings", test_openai_embeddings),
            ("Pinecone Connection", test_pinecone_connection),
            ("Pinecone Operations", test_pinecone_operations)
        ]



def test_environment_variables() -> Tuple[bool, Dict[str, str]]:
    """Test if required environment variables are set."""
    print("Checking environment variables...")
    
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENVIRONMENT_REGION": os.getenv("PINECONE_ENVIRONMENT_REGION"),
    }
    
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if not var_value or var_value.strip() == "":
            missing_vars.append(var_name)
            print(f"FAIL: {var_name}: Not set")
        else:
            # Mask API keys for security
            if "API_KEY" in var_name:
                masked_value = var_value[:8] + "..." + var_value[-4:] if len(var_value) > 12 else "***"
                print(f"PASS: {var_name}: {masked_value}")
            else:
                print(f"PASS: {var_name}: {var_value}")
    
    if missing_vars:
        print(f"\nMissing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all required variables are set.")
        return False, required_vars
    
    print("All environment variables are set!")
    return True, required_vars

def test_openai_connection() -> bool:
    """Test OpenAI API connectivity."""
    print("\nTesting OpenAI connection...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a connection test."}],
            max_tokens=10
        )
        
        if response.choices and response.choices[0].message:
            print("PASS: OpenAI API connection successful!")
            print(f"   Model: gpt-3.5-turbo")
            print(f"   Response: {response.choices[0].message.content[:50]}...")
            return True
        else:
            print("FAIL: OpenAI API returned empty response")
            return False
            
    except ImportError:
        print("FAIL: OpenAI library not installed. Run: pip install openai")
        return False
    except Exception as e:
        print(f"FAIL: OpenAI connection failed: {e}")
        return False

def test_openai_embeddings() -> bool:
    """Test OpenAI embeddings specifically."""
    print("\nTesting OpenAI embeddings...")
    
    try:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            from langchain_community.embeddings import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        # Test embedding generation
        test_text = "This is a test for embedding generation."
        embedding = embeddings.embed_query(test_text)
        
        if embedding and len(embedding) == 1536:
            print("PASS: OpenAI embeddings working correctly!")
            print(f"   Model: {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}")
            print(f"   Dimensions: {len(embedding)} (correct for Pinecone index)")
            return True
        else:
            print(f"FAIL: Embedding dimension mismatch: {len(embedding) if embedding else 0} (expected: 1536)")
            return False
            
    except ImportError as e:
        print(f"FAIL: Required library not installed: {e}")
        return False
    except Exception as e:
        print(f"FAIL: OpenAI embeddings test failed: {e}")
        return False

def test_pinecone_connection() -> bool:
    """Test Pinecone connectivity and index access."""
    print("\nTesting Pinecone connection...")
    
    try:
        from pinecone import Pinecone
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # List indexes to test connection
        indexes = pc.list_indexes()
        print("PASS: Pinecone API connection successful!")
        print(f"   Available indexes: {len(indexes.indexes)}")
        
        # Check if our specific index exists
        index_name = os.getenv("PINECONE_INDEX_NAME", "small-blogs-emmbeddings-index")
        
        index_found = False
        for index in indexes.indexes:
            if index.name == index_name:
                index_found = True
                print(f"PASS: Target index found: {index_name}")
                print(f"   Status: {index.status.state}")
                print(f"   Host: {index.host}")
                print(f"   Dimension: {index.dimension}")
                print(f"   Metric: {index.metric}")
                break
        
        if not index_found:
            print(f"FAIL: Target index not found: {index_name}")
            print("  Available indexes:")
            for index in indexes.indexes:
                print(f"   - {index.name}")
            return False
        
        return True
        
    except ImportError:
        print("FAIL: Pinecone library not installed. Run: pip install pinecone")
        return False
    except Exception as e:
        print(f"FAIL: Pinecone connection failed: {e}")
        return False

def test_pinecone_operations() -> bool:
    """Test basic Pinecone operations."""
    print("\nTesting Pinecone operations...")
    
    try:
        from pinecone import Pinecone
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            from langchain_community.embeddings import OpenAIEmbeddings
        
        # Initialize components
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "small-blogs-emmbeddings-index")
        
        # Get the index directly from Pinecone client
        index = pc.Index(index_name)
        
        # Test basic index operations
        # Create a test vector
        test_embedding = embeddings.embed_query("test query for pinecone operations")
        
        # Test upsert operation with a simple vector
        test_id = "test-vector-setup-check"
        index.upsert(
            vectors=[(test_id, test_embedding, {"test": "setup_check"})]
        )
        print("PASS: Vector upsert successful")
        
        # Test query operation
        query_response = index.query(
            vector=test_embedding,
            top_k=1,
            include_metadata=True
        )
        
        if query_response and 'matches' in query_response:
            print("PASS: Vector query successful")
            print(f"   Found {len(query_response['matches'])} matches")
        
        # Clean up test vector
        try:
            index.delete(ids=[test_id])
            print("PASS: Test vector cleanup successful")
        except:
            print("WARNING: Test vector cleanup skipped (not critical)")
        
        # Test LangChain integration if possible
        try:
            from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
            
            # Try the LangChain integration
            vectorstore = PineconeLangChain.from_existing_index(
                embedding=embeddings,
                index_name=index_name
            )
            
            # Test search functionality
            test_results = vectorstore.similarity_search("housing rights", k=1)
            print("PASS: LangChain integration working!")
            print(f"   Search test returned: {len(test_results)} results")
            
        except Exception as langchain_error:
            print(f"WARNING: LangChain integration issue: {langchain_error}")
            print("   Direct Pinecone operations work, but LangChain wrapper may have issues")
            print("   This might affect some advanced features but core functionality should work")
        
        print("PASS: Pinecone operations working!")
        print(f"   Index connected: {index_name}")
        print(f"   Embedding dimensions: {len(test_embedding)}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Pinecone operations test failed: {e}")
        print("   This could be due to:")
        print("   - Network connectivity issues")
        print("   - Invalid API key or permissions")
        print("   - Index name mismatch")
        print("   - Version compatibility issues")
        return False

def run_dependency_check() -> bool:
    """Check if all required dependencies are installed."""
    print("\nChecking required dependencies...")
    
    required_packages = [
        "streamlit",
        "openai", 
        "pinecone",
        "langchain",
        "langchain_community",
        "dotenv",  # python-dotenv imports as 'dotenv'
        "loguru"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"PASS: {package}: Installed")
        except ImportError:
            missing_packages.append(package)
            print(f"FAIL: {package}: Not installed")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("All required dependencies are installed!")
    return True

def main():
    """Run all setup tests."""
    print("Setup complete")
    print("=" * 50)
    
    start_time = time.time()
    
    # Track test results
    tests = [
        ("Dependencies", run_dependency_check),
        ("Environment Variables", test_environment_variables),
        ("OpenAI Connection", test_openai_connection),
        ("OpenAI Embeddings", test_openai_embeddings),
        ("Pinecone Connection", test_pinecone_connection),
        ("Pinecone Operations", test_pinecone_operations)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if test_name == "Environment Variables":
                success, _ = test_func()
            else:
                success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"FAIL: {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Duration: {time.time() - start_time:.2f} seconds")
    
    if passed == total:
        print("\nAll tests passed.")
        print("You can now start the application with: streamlit run app.py")
        return True
    else:
        print(f"\n{total - passed} test(s) failed. Please fix the issues above before running LAISA.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)