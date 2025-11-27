"""
Calcmate Chatbot Frontend
-------------------------
Flask application for the Calcmate neuro-symbolic math assistant.
Provides a ChatGPT-like interface for mathematical problem solving.
"""

from flask import Flask, render_template, request, jsonify
import json
import time
import sys
import os

# Add the current directory to Python path to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from complete_pipeline_demo import CompletePipelineDemo
    CALCMATE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Calcmate pipeline not available: {e}")
    CALCMATE_AVAILABLE = False

app = Flask(__name__)

class CalcmateChatbot:
    """Chatbot interface for Calcmate pipeline"""
    
    def __init__(self):
        self.pipeline = None
        self.chat_history = []
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize the Calcmate pipeline"""
        if not CALCMATE_AVAILABLE:
            return False
            
        try:
            print("🚀 Initializing Calcmate Pipeline...")
            self.pipeline = CompletePipelineDemo()
            success = self.pipeline.load_complete_system()
            if success:
                print("✅ Calcmate Pipeline initialized successfully!")
            else:
                print("❌ Failed to initialize Calcmate Pipeline")
            return success
        except Exception as e:
            print(f"❌ Error initializing Calcmate: {e}")
            return False
    
    def process_message(self, message):
        """Process a user message through the Calcmate pipeline"""
        if not self.pipeline or not CALCMATE_AVAILABLE:
            return {
                'type': 'error',
                'content': 'Calcmate pipeline is not available. Please check the backend setup.',
                'timestamp': time.time()
            }
        
        try:
            print(f"🔍 Processing query: {message}")
            
            # Run through pipeline
            start_time = time.time()
            result = self.pipeline.pipeline(message, top_k=3, explain=True)
            processing_time = time.time() - start_time
            
            # Format response
            response = self._format_pipeline_response(result, message, processing_time)
            
            # Add to chat history
            self.chat_history.append({
                'user': message,
                'assistant': response,
                'timestamp': time.time()
            })
            
            return response
            
        except Exception as e:
            error_response = {
                'type': 'error',
                'content': f'Sorry, I encountered an error while processing your question: {str(e)}',
                'timestamp': time.time()
            }
            print(f"❌ Pipeline error: {e}")
            return error_response
    
    def _format_pipeline_response(self, result, original_query, processing_time):
        """Format the pipeline result into a chat response"""
        response = {
            'type': 'success',
            'content': '',
            'details': {},
            'processing_time': f"{processing_time:.2f}s",
            'timestamp': time.time()
        }
        
        # Basic solution
        solution = getattr(result, 'solution', None)
        if solution:
            solution_text = "**Solution:**\n"
            for var, val in solution.items():
                solution_text += f"- {var} = {val}\n"
            response['content'] += solution_text + "\n"
        
        # If no solution found, provide helpful message
        if not solution:
            response['content'] = "I analyzed your problem but couldn't find a definitive solution. Here's what I found:\n\n"
        
        # Result type
        result_type = getattr(result, 'result_type', 'unknown')
        response['details']['method'] = result_type.upper()
        
        # Retrieved similar problems
        results = getattr(result, 'results', [])
        if results:
            response['details']['similar_problems_found'] = len(results)
            similar_text = "**Similar problems I found:**\n"
            for i, res in enumerate(results[:2], 1):  # Show top 2
                text = res.get('text', '')[:100] + "..." if len(res.get('text', '')) > 100 else res.get('text', '')
                similarity = res.get('similarity', 0)
                similar_text += f"{i}. (Similarity: {similarity:.3f}) {text}\n"
            response['content'] += similar_text + "\n"
        
        # Equations found
        equations = getattr(result, 'equations', None)
        if equations:
            eq_text = "**Equations identified:**\n"
            for eq in equations:
                eq_text += f"- {eq}\n"
            response['content'] += eq_text + "\n"
        
        # LLM Reasoning
        reasoning = getattr(result, 'reasoning', None)
        if reasoning:
            response['details']['reasoning'] = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
        
        # Add note if available
        note = getattr(result, 'note', None)
        if note:
            response['content'] += f"**Note:** {note}\n"
        
        # If content is still empty, provide fallback
        if not response['content']:
            response['content'] = f"I received your question '{original_query}' but couldn't generate a detailed response. The problem might be too complex or outside my current capabilities."
        
        return response
    
    def get_chat_history(self):
        """Get the chat history"""
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        return True

# Initialize chatbot
chatbot = CalcmateChatbot()

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html', pipeline_available=CALCMATE_AVAILABLE)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Process the message
        response = chatbot.process_message(message)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'type': 'error',
            'content': f'Internal server error: {str(e)}',
            'timestamp': time.time()
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history"""
    return jsonify({'history': chatbot.get_chat_history()})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear chat history"""
    success = chatbot.clear_history()
    return jsonify({'success': success})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    status = {
        'pipeline_available': CALCMATE_AVAILABLE,
        'pipeline_initialized': chatbot.pipeline is not None,
        'system_ready': CALCMATE_AVAILABLE and chatbot.pipeline is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Starting Calcmate Chatbot Server...")
    print("📍 Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)