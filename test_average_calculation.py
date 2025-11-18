"""
Test script to verify the average calculation functionality
"""

import asyncio
from chainlit_chatbot import CalcMateChatbot


async def test_average_calculation():
    """Test the average calculation functionality"""
    print("🧪 Testing Average Calculation")
    print("=" * 40)
    
    chatbot = CalcMateChatbot()
    
    # Test the simple arithmetic function directly
    test_queries = [
        "what is the average of 10, 5 and 25",
        "find the mean of 10, 5, 25",
        "calculate the average of 10, 5, 25",
        "what is the sum of 10, 5 and 25",
        "find the product of 2, 3 and 4"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        # Test the simple arithmetic function
        result = chatbot._try_simple_arithmetic(query)
        
        if result:
            print(f"✅ Success!")
            print(f"   Solution: {result.solution}")
            print(f"   Type: {result.result_type}")
            print(f"   Equation: {result.equations[0] if result.equations else 'N/A'}")
            print(f"   Reasoning: {result.reasoning}")
        else:
            print("❌ No simple arithmetic solution found")
    
    print("\n🎉 Average calculation test completed!")


if __name__ == "__main__":
    asyncio.run(test_average_calculation())
