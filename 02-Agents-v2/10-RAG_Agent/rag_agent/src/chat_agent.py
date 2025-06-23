"""
Chat Agent using LangGraph for document-based conversations
"""

from typing import Dict, Any, List
from langgraph.graph import Graph, StateGraph, END
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from .vector_store import VectorStoreManager

class ChatState:
    """State management for chat conversations"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.context: str = ""
        self.query: str = ""
        self.response: str = ""

class ChatAgent:
    """Main chat agent using LangGraph"""
    
    def __init__(self, model_name: str = "llama3.2", vector_store: VectorStoreManager = None):
        self.model_name = model_name
        self.vector_store = vector_store
        self.llm = ChatOllama(model=model_name, temperature=0.7)
        self.conversation_history = []
        
        # Create the LangGraph workflow
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve_context")
        
        return workflow.compile()
    
    def _retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context from vector store"""
        query = state.get("query", "")
        
        if self.vector_store and query:
            # Search for relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            state["context"] = context
        else:
            state["context"] = ""
        
        return state
    
    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using LLM"""
        query = state.get("query", "")
        context = state.get("context", "")
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt())
        ])
        
        # Create chain
        chain = (
            RunnablePassthrough()
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Generate response
        response = chain.invoke({
            "context": context,
            "question": query,
            "chat_history": self._format_chat_history()
        })
        
        state["response"] = response
        return state
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the LLM"""
        return """You are a helpful AI assistant that answers questions based on the provided context from documents. 

Guidelines:
- Use the provided context to answer questions accurately
- If the context doesn't contain enough information, say so clearly
- Provide detailed explanations when helpful
- For code-related questions, provide examples when possible
- Maintain a conversational and helpful tone
- Reference specific parts of the documents when relevant

Context: {context}

Chat History: {chat_history}"""
    
    def _get_human_prompt(self) -> str:
        """Get human prompt template"""
        return "Question: {question}"
    
    def _format_chat_history(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted_history = []
        for entry in self.conversation_history[-5:]:  # Last 5 exchanges
            formatted_history.append(f"Human: {entry['human']}")
            formatted_history.append(f"Assistant: {entry['assistant']}")
        
        return "\n".join(formatted_history)
    
    def chat(self, query: str) -> str:
        """Main chat method"""
        try:
            # Prepare initial state
            initial_state = {
                "query": query,
                "context": "",
                "response": ""
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            response = result.get("response", "I'm sorry, I couldn't generate a response.")
            
            # Update conversation history
            self.conversation_history.append({
                "human": query,
                "assistant": response
            })
            
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response
            
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []