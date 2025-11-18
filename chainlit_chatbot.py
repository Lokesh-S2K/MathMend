"""
CalcMate Chainlit Chatbot Interface
----------------------------------
A simple chatbot UI for the complete neuro-symbolic pipeline system.
Users can interact with the mathematical reasoning system through a web interface.
"""

import chainlit as cl
import time
import json
import asyncio
from typing import Dict, Any, List
from dspy_modules import (
    SmartRetrievalPipeline, 
    SymbolicSolver, 
    Verifier, 
    LLMReasoner,
    initialize_ctransformers_model
)
from pipeline_sequence.embedder import encode_texts
from sympy import symbols, Eq


class CalcMateChatbot:
    """Chatbot interface for the neuro-symbolic mathematical reasoning system"""
    
    def __init__(self):
        self.pipeline = None
        self.llm_model = None
        self.symbolic_solver = SymbolicSolver()
        self.verifier = Verifier()
        self.llm_reasoner = None
        self.index_path = "output/embeddings/faiss_index_20251015_145706.bin"
        self.idmap_path = "output/embeddings/faiss_id_map_20251015_145706.json"
        self.is_initialized = False
        
        # User session tracking
        self.user_stats = {
            'total_queries': 0,
            'symbolic_solutions': 0,
            'llm_solutions': 0,
            'unresolved': 0,
            'total_time': 0
        }

    async def initialize_system(self):
        """Initialize the complete neuro-symbolic system"""
        if self.is_initialized:
            return True
            
        try:
            # Show loading message
            await cl.Message(
                content="🔄 Initializing CalcMate Neuro-Symbolic System...\n"
                       "This may take a few moments on first run.",
                author="System"
            ).send()
            
            # Initialize CTransformers LLM
            await cl.Message(
                content="🤖 Loading CTransformers LLM (Llama-2-7B)...",
                author="System"
            ).send()
            
            self.llm_model = initialize_ctransformers_model()
            
            if self.llm_model is not None:
                await cl.Message(
                    content="✅ LLM loaded successfully!",
                    author="System"
                ).send()
            else:
                await cl.Message(
                    content="⚠️ LLM not available, using fallback methods",
                    author="System"
                ).send()
            
            # Load main pipeline
            await cl.Message(
                content="🔄 Loading mathematical knowledge base...",
                author="System"
            ).send()
            
            self.pipeline = SmartRetrievalPipeline(
                self.index_path, 
                self.idmap_path,
                llm_model=self.llm_model
            )
            
            # Initialize LLM reasoner
            self.llm_reasoner = LLMReasoner(self.llm_model)
            
            self.is_initialized = True
            
            await cl.Message(
                content="🎉 **CalcMate System Ready!**\n\n"
                       "I can help you with:\n"
                       "• Mathematical problem solving\n"
                       "• Equation solving\n"
                       "• Step-by-step reasoning\n"
                       "• Finding similar problems\n\n"
                       "Just ask me any math question!",
                author="CalcMate"
            ).send()
            
            return True
            
        except Exception as e:
            await cl.Message(
                content=f"❌ Failed to initialize system: {str(e)}",
                author="System"
            ).send()
            return False

    async def process_math_query(self, query: str) -> Dict[str, Any]:
        """Process a mathematical query through the complete pipeline"""
        start_time = time.time()
        
        try:
            # First, try simple arithmetic calculations
            simple_result = self._try_simple_arithmetic(query)
            if simple_result:
                total_time = time.time() - start_time
                self.user_stats['total_queries'] += 1
                self.user_stats['total_time'] += total_time
                self.user_stats['symbolic_solutions'] += 1
                
                return {
                    'result': simple_result,
                    'processing_time': total_time,
                    'result_type': 'arithmetic',
                    'success': True
                }
            
            # Run through the complete pipeline
            result = self.pipeline(query, top_k=3, explain=True)
            
            total_time = time.time() - start_time
            
            # Update user stats
            self.user_stats['total_queries'] += 1
            self.user_stats['total_time'] += total_time
            
            result_type = getattr(result, 'result_type', 'unknown')
            if result_type == 'symbolic':
                self.user_stats['symbolic_solutions'] += 1
            elif result_type == 'llm':
                self.user_stats['llm_solutions'] += 1
            else:
                self.user_stats['unresolved'] += 1
            
            return {
                'result': result,
                'processing_time': total_time,
                'result_type': result_type,
                'success': True
            }
            
        except Exception as e:
            return {
                'result': None,
                'processing_time': time.time() - start_time,
                'result_type': 'error',
                'success': False,
                'error': str(e)
            }

    def _try_simple_arithmetic(self, query: str) -> Any:
        """Try to solve simple arithmetic problems directly"""
        import re
        import statistics
        
        query_lower = query.lower().strip()
        
        # Extract numbers from the query
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', query)
        if not numbers:
            return None
        
        numbers = [float(n) for n in numbers]
        
        # Handle average/mean calculations
        if any(word in query_lower for word in ['average', 'mean']):
            if len(numbers) >= 2:
                avg = statistics.mean(numbers)
                # Create a simple result object
                class SimpleResult:
                    def __init__(self, solution, result_type="arithmetic"):
                        self.solution = solution
                        self.result_type = result_type
                        self.equations = [f"average = ({' + '.join(map(str, numbers))}) / {len(numbers)}"]
                        self.reasoning = f"To find the average of {len(numbers)} numbers, add them together and divide by {len(numbers)}."
                        self.results = []
                        self.note = f"Simple arithmetic calculation: average of {numbers}"
                
                return SimpleResult(avg)
        
        # Handle sum calculations
        elif any(word in query_lower for word in ['sum', 'total', 'add']):
            if len(numbers) >= 2:
                total = sum(numbers)
                class SimpleResult:
                    def __init__(self, solution, result_type="arithmetic"):
                        self.solution = solution
                        self.result_type = result_type
                        self.equations = [f"sum = {' + '.join(map(str, numbers))}"]
                        self.reasoning = f"To find the sum, add all the numbers together."
                        self.results = []
                        self.note = f"Simple arithmetic calculation: sum of {numbers}"
                
                return SimpleResult(total)
        
        # Handle product calculations
        elif any(word in query_lower for word in ['product', 'multiply']):
            if len(numbers) >= 2:
                product = 1
                for num in numbers:
                    product *= num
                class SimpleResult:
                    def __init__(self, solution, result_type="arithmetic"):
                        self.solution = solution
                        self.result_type = result_type
                        self.equations = [f"product = {' * '.join(map(str, numbers))}"]
                        self.reasoning = f"To find the product, multiply all the numbers together."
                        self.results = []
                        self.note = f"Simple arithmetic calculation: product of {numbers}"
                
                return SimpleResult(product)
        
        return None

    async def format_response(self, query: str, pipeline_result: Dict[str, Any]) -> str:
        """Format the pipeline result into a user-friendly response"""
        if not pipeline_result['success']:
            return f"❌ **Error Processing Query**\n\nI encountered an error: {pipeline_result.get('error', 'Unknown error')}"
        
        result = pipeline_result['result']
        result_type = pipeline_result['result_type']
        processing_time = pipeline_result['processing_time']
        
        response_parts = []
        
        # Header with processing info
        response_parts.append(f"⏱ **Processing Time:** {processing_time:.2f} seconds")
        response_parts.append(f"🎯 **Solution Method:** {result_type.title()}")
        response_parts.append("")
        
        # Show retrieved similar problems
        retrieved_results = getattr(result, 'results', [])
        if retrieved_results:
            response_parts.append("📋 **Similar Problems Found:**")
            for i, res in enumerate(retrieved_results[:3], 1):
                similarity = res.get('similarity', 0)
                text = res.get('text', '')[:150] + "..."
                response_parts.append(f"{i}. *Similarity: {similarity:.3f}*")
                response_parts.append(f"   {text}")
                response_parts.append("")
        
        # Show extracted equations
        equations = getattr(result, 'equations', None)
        if equations:
            response_parts.append("📘 **Extracted Equations:**")
            for eq in equations:
                response_parts.append(f"• {eq}")
            response_parts.append("")
        
        # Show solution
        solution = getattr(result, 'solution', None)
        if solution:
            response_parts.append("🧮 **Solution:**")
            if isinstance(solution, dict):
                for var, val in solution.items():
                    response_parts.append(f"**{var} = {val}**")
            else:
                # Handle non-dictionary solutions (e.g., single values)
                response_parts.append(f"**Answer: {solution}**")
            response_parts.append("")
            
            # Show verification residuals
            residuals = getattr(result, 'residuals', None)
            if residuals:
                response_parts.append("🔍 **Verification:**")
                for eq, val in residuals.items():
                    if isinstance(val, dict):
                        satisfied = val.get('satisfied', False)
                        value = val.get('value', 'N/A')
                        status = "✅" if satisfied else "❌"
                        response_parts.append(f"{status} {eq}: {value}")
                    else:
                        response_parts.append(f"• {eq}: {val}")
                response_parts.append("")
        
        # Show LLM reasoning
        reasoning = getattr(result, 'reasoning', None)
        if reasoning:
            response_parts.append("🤖 **Step-by-Step Reasoning:**")
            # Truncate very long reasoning
            display_reasoning = reasoning[:800] + "..." if len(reasoning) > 800 else reasoning
            response_parts.append(display_reasoning)
            response_parts.append("")
        
        # Show note if unresolved
        note = getattr(result, 'note', None)
        if note:
            response_parts.append(f"ℹ️ **Note:** {note}")
        
        return "\n".join(response_parts)

    async def show_user_stats(self):
        """Show user statistics"""
        if self.user_stats['total_queries'] == 0:
            await cl.Message(
                content="📊 **No queries processed yet.**\nAsk me a math question to see your stats!",
                author="CalcMate"
            ).send()
            return
        
        total = self.user_stats['total_queries']
        avg_time = self.user_stats['total_time'] / total
        
        stats_text = f"""📊 **Your Session Statistics:**

🔢 **Total Queries:** {total}
⏱ **Average Processing Time:** {avg_time:.2f} seconds
🧮 **Symbolic Solutions:** {self.user_stats['symbolic_solutions']} ({self.user_stats['symbolic_solutions']/total*100:.1f}%)
🤖 **LLM Solutions:** {self.user_stats['llm_solutions']} ({self.user_stats['llm_solutions']/total*100:.1f}%)
❓ **Unresolved:** {self.user_stats['unresolved']} ({self.user_stats['unresolved']/total*100:.1f}%)

💡 **System Status:** All components operational!"""
        
        await cl.Message(
            content=stats_text,
            author="CalcMate"
        ).send()


# Global chatbot instance
chatbot = CalcMateChatbot()


@cl.on_chat_start
async def start():
    """Initialize the chatbot when a new chat session starts"""
    await cl.Message(
        content="👋 **Welcome to CalcMate!**\n\n"
               "I'm your AI mathematical reasoning assistant powered by:\n"
               "• Neural similarity search\n"
               "• Symbolic equation solving\n"
               "• CTransformers LLM reasoning\n"
               "• Solution verification\n\n"
               "Initializing system...",
        author="CalcMate"
    ).send()
    
    # Initialize the system
    success = await chatbot.initialize_system()
    
    if not success:
        await cl.Message(
            content="❌ **System Initialization Failed**\n\n"
                   "Please check that all required files are present and try again.",
            author="System"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    query = message.content.strip()
    
    if not query:
        await cl.Message(
            content="Please enter a mathematical problem or question!",
            author="CalcMate"
        ).send()
        return
    
    # Handle special commands
    if query.lower() in ['/help', '/h']:
        await cl.Message(
            content="""📚 **CalcMate Help**

**What I can do:**
• Solve mathematical equations
• Provide step-by-step reasoning
• Find similar problems
• Verify solutions

**Example questions:**
• "Two numbers sum to 50 and their difference is 10. What are the numbers?"
• "A car travels 120 km in 2 hours. What is its speed?"
• "If a pizza costs $18 and you order 3 pizzas, what's the total cost?"
• "A rectangle has length 12 cm and width 8 cm. What is its area?"

**Commands:**
• `/help` - Show this help message
• `/stats` - Show your session statistics
• `/examples` - Show example problems

Just ask me any math question!""",
            author="CalcMate"
        ).send()
        return
    
    if query.lower() in ['/stats', '/statistics']:
        await chatbot.show_user_stats()
        return
    
    if query.lower() in ['/examples', '/example']:
        await cl.Message(
            content="""📝 **Example Math Problems to Try:**

**Algebra:**
• "Two numbers sum to 50 and their difference is 10. What are the numbers?"
• "A number is 5 more than twice another number. If their sum is 35, what are the numbers?"

**Speed/Distance/Time:**
• "A car travels 120 km in 2 hours. What is its speed?"
• "A train travels at 60 km/h. How far does it travel in 2.5 hours?"

**Cost/Price:**
• "If a pizza costs $18 and you order 3 pizzas, what's the total cost?"
• "A store offers 20% discount on a $50 item. What's the final price?"

**Geometry:**
• "A rectangle has length 12 cm and width 8 cm. What is its area?"
• "A circle has radius 7 cm. What is its circumference?"

**Complex Problems:**
• "A train leaves station A at 9 AM traveling at 60 km/h. Another train leaves station B at 10 AM traveling at 80 km/h. If the stations are 300 km apart, when will they meet?"

Try any of these or ask your own question!""",
            author="CalcMate"
        ).send()
        return
    
    # Process the mathematical query
    await cl.Message(
        content="🔄 Processing your mathematical query...",
        author="CalcMate"
    ).send()
    
    # Process through the pipeline
    pipeline_result = await chatbot.process_math_query(query)
    
    # Format and send response
    response = await chatbot.format_response(query, pipeline_result)
    
    await cl.Message(
        content=response,
        author="CalcMate"
    ).send()


@cl.on_stop
async def on_stop():
    """Handle when the chat session stops"""
    await cl.Message(
        content="👋 **Thank you for using CalcMate!**\n\n"
               "Your mathematical reasoning session has ended. "
               "Feel free to start a new session anytime!",
        author="CalcMate"
    ).send()


if __name__ == "__main__":
    # This will be handled by Chainlit
    pass
