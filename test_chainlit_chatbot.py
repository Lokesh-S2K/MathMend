"""
Test script for the Chainlit CalcMate Chatbot
---------------------------------------------
This script demonstrates how to test the chatbot interface programmatically.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch
from chainlit_chatbot import CalcMateChatbot
import pytest


@pytest.mark.asyncio
async def test_chatbot_functionality():
    """Test the chatbot functionality programmatically"""
    print("🧪 Testing CalcMate Chatbot Functionality")
    print("=" * 50)
    
    with patch('chainlit_chatbot.cl.Message') as mock_message_class:
        mock_message_instance = AsyncMock()
        mock_message_instance.send = AsyncMock()
        mock_message_class.return_value = mock_message_instance
        # Initialize chatbot
        chatbot = CalcMateChatbot()
    
    # Test system initialization
        print("1. Testing system initialization...")
        success = await chatbot.initialize_system()
        if success:
            print("✅ System initialized successfully!")
        else:
            print("❌ System initialization failed!")
            return
        
        # Test mathematical queries
        test_cases = [
                {
                    "query": "Two numbers sum to 50 and their difference is 10. What are the numbers?",
                    "expected_answer": None # This will be handled by the pipeline, not simple arithmetic
                },
                {
                    "query": "A car travels 120 km in 2 hours. What is its speed?",
                    "expected_answer": 60.0 # km/h
                },
                {
                    "query": "If a pizza costs $18 and you order 3 pizzas, what's the total cost?",
                    "expected_answer": 54.0 # dollars
                },
            ]
        
        print("\n2. Testing mathematical queries...")
        for i, test_case in enumerate(test_cases, 1):
            query = test_case["query"]
            expected_answer = test_case["expected_answer"]
            
            print(f"\n[{i}] Testing: {query}")
            print("-" * 40)
            
            # Process query
            result = await chatbot.process_math_query(query)
            
            if result['success']:
                print(f"✅ Success! Processing time: {result['processing_time']:.2f}s")
                print(f"   Result type: {result['result_type']}")
                print(f"   Full result dictionary: {result}") # Added this line to print the full result
                
                # Format response
                response = await chatbot.format_response(query, result)
                print(f"   Response preview: {response[:200]}...")
                
                # Assert numerical correctness
                actual_answer = result.get('answer') # Assuming the answer is in result['answer']
                print(f"   Expected: {expected_answer}, Actual: {actual_answer}")
                assert actual_answer == expected_answer, f"❌ Incorrect answer for query: {query}. Expected {expected_answer}, got {actual_answer}"
                print(f"   ✅ Answer is numerically correct!")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Test user stats
        print("\n3. Testing user statistics...")
        await chatbot.show_user_stats()
        
        print("\n🎉 Chatbot functionality test completed!")


def print_usage_instructions():
    """Print usage instructions for the Chainlit chatbot"""
    print("\n" + "=" * 60)
    print("🎯 CALCMATE CHAINLIT CHATBOT USAGE INSTRUCTIONS")
    print("=" * 60)
    print()
    print("🚀 **Starting the Chatbot:**")
    print("   chainlit run chainlit_chatbot.py")
    print()
    print("🌐 **Accessing the Interface:**")
    print("   Open your browser and go to: http://localhost:8000")
    print()
    print("💬 **Using the Chatbot:**")
    print("   • Type any mathematical problem or question")
    print("   • Use commands like /help, /stats, /examples")
    print("   • The system will process your query through the complete pipeline")
    print()
    print("🔧 **Available Commands:**")
    print("   /help     - Show help information")
    print("   /stats    - Show your session statistics")
    print("   /examples - Show example problems to try")
    print()
    print("📝 **Example Questions to Try:**")
    print("   • 'Two numbers sum to 50 and their difference is 10. What are the numbers?'")
    print("   • 'A car travels 120 km in 2 hours. What is its speed?'")
    print("   • 'If a pizza costs $18 and you order 3 pizzas, what's the total cost?'")
    print("   • 'A rectangle has length 12 cm and width 8 cm. What is its area?'")
    print()
    print("🎯 **What the Chatbot Does:**")
    print("   • Retrieves similar mathematical problems from knowledge base")
    print("   • Extracts equations from natural language")
    print("   • Solves problems symbolically using SymPy")
    print("   • Uses CTransformers LLM for complex reasoning")
    print("   • Verifies solutions and shows step-by-step reasoning")
    print("   • Tracks your session statistics")
    print()
    print("⚡ **Features:**")
    print("   • Real-time processing with progress indicators")
    print("   • Detailed explanations and reasoning steps")
    print("   • Similar problem suggestions")
    print("   • Solution verification with residuals")
    print("   • Session statistics and performance tracking")
    print()
    print("🛑 **Stopping the Server:**")
    print("   Press Ctrl+C in the terminal where chainlit is running")
    print()
    print("=" * 60)


if __name__ == "__main__":
    print_usage_instructions()
    
    # Ask if user wants to run the test
    try:
        response = input("\n🧪 Would you like to run the functionality test? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            asyncio.run(test_chatbot_functionality())
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")
    
    print("\n🎉 Ready to use the Chainlit chatbot!")
    print("Run: chainlit run chainlit_chatbot.py")
