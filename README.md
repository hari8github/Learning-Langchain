# 🔍 AI Search Assistant - Perplexity Clone

A modern AI-powered search assistant that mimics Perplexity's functionality, built with LangChain, Groq, and real-time streaming responses.

## ✨ Features

- **🌐 Real-time Web Search** - Live search results using SerpAPI
- **🧮 Mathematical Operations** - Built-in calculator with advanced functions
- **⚡ Streaming Responses** - Real-time token streaming via Groq API
- **🔧 Transparent Process** - Shows step-by-step agent reasoning
- **🎯 Tool Tracking** - Displays which tools were used for each answer
- **📱 Responsive Design** - Clean, minimalist interface inspired by Perplexity

## 🏗️ Architecture

### Backend (FastAPI + LangChain)
- **LangChain Agents** - Intelligent tool selection and orchestration
- **Custom Tools** - Web search, mathematical operations, final answer compilation
- **Streaming Support** - Real-time response streaming
- **Tool Execution** - Async tool execution with error handling

### Frontend (Next.js + TypeScript)
- **Real-time Updates** - Live step visualization during agent execution
- **Markdown Rendering** - Rich text display for answers
- **Incomplete JSON Parser** - Handles streaming JSON responses
- **Responsive UI** - Mobile-friendly design

## 🛠️ Tech Stack

**Backend:**
- FastAPI
- LangChain
- Groq API (LLaMA 3-8B)
- SerpAPI
- Python AsyncIO
- Pydantic

**Frontend:**
- Next.js 14
- TypeScript
- Tailwind CSS
- React Markdown
- Incomplete JSON Parser

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Environment Variables

Create a `.env` file in the `api` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

### Backend Setup

1. Navigate to the API directory:
```bash
cd api
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the app directory:
```bash
cd app
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 🔧 API Endpoints

### `POST /invoke`
Main endpoint for processing queries with streaming responses.

**Request:**
```json
{
  "content": "Your question here"
}
```

**Response:** Streaming text with step information and final answer.

### `GET /health`
Health check endpoint.

### `POST /test` & `POST /test-stream`
Testing endpoints for debugging.

## 🎯 How It Works

1. **User Input** - User submits a question through the web interface
2. **Agent Processing** - LangChain agent analyzes the query and selects appropriate tools
3. **Tool Execution** - Agent executes tools (search, calculate, etc.) as needed
4. **Streaming Response** - Results are streamed back in real-time
5. **Final Answer** - Compiled answer with source attribution and tool tracking

## 🔍 Available Tools

- **search(query)** - Web search using SerpAPI
- **add(x, y)** - Addition operation
- **subtract(x, y)** - Subtraction operation
- **multiply(x, y)** - Multiplication operation
- **exponentiate(x, y)** - Exponentiation operation
- **final_answer(answer, tools_used)** - Final response compilation

## 📁 Project Structure

```
├── api/                    # Backend FastAPI application
│   ├── agent.py           # LangChain agent and tools definition
│   ├── main.py            # FastAPI server and endpoints
│   ├── testing.py         # API testing utilities
│   └── .env              # Environment variables
│
├── app/                   # Frontend Next.js application
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx   # Main page component
│   │   │   └── layout.tsx # App layout
│   │   ├── components/
│   │   │   ├── Output.tsx     # Chat output display
│   │   │   ├── TextArea.tsx   # Input component
│   │   │   └── MarkdownRenderer.tsx
│   │   └── types.ts       # TypeScript type definitions
│   └── package.json
```

## 🧪 Example Queries

Try asking:
- "Latest news about artificial intelligence"
- "Calculate compound interest for $1000 at 5% for 3 years"
- "What is the weather in Tokyo today?"
- "Explain quantum computing"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is for educational purposes, inspired by Perplexity AI.

## 🙏 Acknowledgments

- Built following James Briggs' LangChain tutorial
- Inspired by Perplexity AI's interface and functionality
- Uses Groq's lightning-fast LLaMA inference
- Powered by SerpAPI for real-time search results

---
