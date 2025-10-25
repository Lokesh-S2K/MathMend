# 🎯 CalcMate Chainlit Chatbot Interface

A simple, interactive web-based chatbot interface for the complete neuro-symbolic mathematical reasoning system. This interface provides a user-friendly way to interact with the advanced mathematical problem-solving pipeline through a modern web UI.

## 🚀 Quick Start

### 1. Start the Chatbot Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the Chainlit chatbot
chainlit run chainlit_chatbot.py
```

### 2. Access the Interface

Open your browser and navigate to: **http://localhost:8000**

### 3. Start Chatting!

Type any mathematical problem or question and watch the system process it through the complete neuro-symbolic pipeline.

## 🎯 What the Chatbot Does

The CalcMate chatbot provides a complete mathematical reasoning system that:

- **🔍 Retrieves Similar Problems**: Finds mathematically similar problems from the knowledge base
- **📘 Extracts Equations**: Converts natural language to mathematical equations
- **🧮 Solves Symbolically**: Uses SymPy for exact mathematical solutions
- **🤖 Reasons with LLM**: Uses CTransformers (Llama-2-7B) for complex reasoning
- **✅ Verifies Solutions**: Checks solutions against original equations
- **📊 Tracks Statistics**: Monitors your session performance

## 💬 Available Commands

| Command     | Description                                  |
| ----------- | -------------------------------------------- |
| `/help`     | Show help information and available commands |
| `/stats`    | Display your session statistics              |
| `/examples` | Show example problems to try                 |

## 📝 Example Questions

### Algebra Problems

- "Two numbers sum to 50 and their difference is 10. What are the numbers?"
- "A number is 5 more than twice another number. If their sum is 35, what are the numbers?"

### Speed/Distance/Time

- "A car travels 120 km in 2 hours. What is its speed?"
- "A train travels at 60 km/h. How far does it travel in 2.5 hours?"

### Cost/Price Calculations

- "If a pizza costs $18 and you order 3 pizzas, what's the total cost?"
- "A store offers 20% discount on a $50 item. What's the final price?"

### Geometry

- "A rectangle has length 12 cm and width 8 cm. What is its area?"
- "A circle has radius 7 cm. What is its circumference?"

### Complex Word Problems

- "A train leaves station A at 9 AM traveling at 60 km/h. Another train leaves station B at 10 AM traveling at 80 km/h. If the stations are 300 km apart, when will they meet?"

## 🎨 Interface Features

### Real-time Processing

- **Progress Indicators**: Shows system loading and processing status
- **Processing Time**: Displays how long each query takes to process
- **Solution Method**: Indicates whether symbolic or LLM reasoning was used

### Detailed Results

- **Similar Problems**: Shows retrieved problems from the knowledge base
- **Extracted Equations**: Displays mathematical equations extracted from text
- **Step-by-Step Reasoning**: Shows LLM reasoning process
- **Solution Verification**: Displays verification results with residuals
- **Session Statistics**: Tracks your usage and performance

### User Experience

- **Clean Interface**: Modern, responsive web design
- **Interactive Chat**: Natural conversation flow
- **Help System**: Built-in help and examples
- **Error Handling**: Graceful error messages and recovery

## 🔧 Technical Architecture

### Backend Components

- **SmartRetrievalPipeline**: Main pipeline orchestrating all components
- **SymbolicSolver**: SymPy-based equation solving
- **LLMReasoner**: CTransformers LLM for complex reasoning
- **Verifier**: Solution verification and validation
- **FaissIndexer**: Efficient similarity search

### Frontend Components

- **Chainlit Framework**: Modern web UI framework
- **Real-time Updates**: WebSocket-based communication
- **Responsive Design**: Works on desktop and mobile
- **Interactive Elements**: Buttons, progress bars, and formatted output

## 📊 System Performance

### Processing Times

- **Symbolic Solutions**: ~0.003 seconds (very fast)
- **LLM Solutions**: ~20-25 seconds (comprehensive reasoning)
- **Retrieval**: ~0.1 seconds (efficient similarity search)

### Solution Distribution

- **Symbolic**: For problems with clear mathematical equations
- **LLM**: For complex word problems requiring reasoning
- **Hybrid**: Combines both approaches for optimal results

## 🛠️ Development and Testing

### Test the Chatbot

```bash
# Run the test script
python test_chainlit_chatbot.py
```

### Customize the Interface

The chatbot interface can be customized by modifying:

- `chainlit_chatbot.py`: Main chatbot logic and UI
- `dspy_modules.py`: Core mathematical reasoning modules
- Response formatting and styling

### Add New Features

- New mathematical problem types
- Additional LLM models
- Enhanced verification methods
- Custom UI components

## 🚨 Troubleshooting

### Common Issues

**1. Server Won't Start**

```bash
# Check if port 8000 is available
lsof -i :8000

# Try a different port
chainlit run chainlit_chatbot.py --port 8001
```

**2. LLM Not Loading**

- Ensure CTransformers model is downloaded
- Check available disk space
- Verify model path in `dspy_modules.py`

**3. Pipeline Errors**

- Verify FAISS index files exist in `output/embeddings/`
- Check that all dependencies are installed
- Review error logs in the terminal

### Getting Help

- Check the `/help` command in the chatbot
- Review the console output for error messages
- Ensure all required files are present

## 🎉 Success Indicators

When everything is working correctly, you should see:

✅ **System Initialization**

- "CalcMate System Ready!" message
- All components loaded successfully
- LLM model initialized

✅ **Query Processing**

- Real-time progress indicators
- Detailed results with explanations
- Similar problems retrieved
- Solutions verified

✅ **User Experience**

- Fast response times for symbolic problems
- Comprehensive reasoning for complex problems
- Clear, formatted output
- Session statistics tracking

## 🔮 Future Enhancements

Potential improvements for the chatbot:

- **Multi-language Support**: Support for different languages
- **Voice Input**: Speech-to-text for mathematical problems
- **Graphical Output**: Visual representation of solutions
- **Collaborative Features**: Share problems and solutions
- **Advanced Analytics**: Detailed performance metrics
- **Mobile App**: Native mobile application
- **API Integration**: REST API for external applications

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Verify that all required files are present

---

**🎯 The CalcMate Chainlit Chatbot provides a complete, interactive interface to the advanced neuro-symbolic mathematical reasoning system, making complex mathematical problem-solving accessible through a simple, user-friendly web interface!**
