import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import json

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.tools import Tool
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone

import prompt_template as prompt_template

load_dotenv()

# Configuration constants
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "small-blogs-emmbeddings-index"
CHAT_TEMPERATURE = 0
DEFAULT_VERBOSE = False

# Agent Templates
LANGUAGE_DETECTION_TEMPLATE = """
You are a language detection specialist. Your task is to identify the language of user input.

Analyze the following text and return ONLY a JSON response with the detected language:
{"language": "language_code", "confidence": confidence_score}

Where language_code is the ISO 639-1 code (e.g., "en", "es", "fr", "de", "nl", "ar") and confidence_score is between 0 and 1.

Text to analyze: {text}

Respond with only the JSON, no additional text.
"""

HOUSING_ASSISTANT_TEMPLATE = """
You are Richard, a knowledgeable housing law specialist working for SIIP. You help users understand housing-related laws, regulations, rights, and responsibilities.

Your expertise includes:
- Tenant rights and landlord obligations
- Housing regulations and compliance
- Rental agreements and lease terms
- Property management laws
- Housing discrimination issues
- Eviction processes and protections

Guidelines:
- Provide clear, accurate information based on the context provided
- If you don't know something, say so clearly
- Avoid giving direct legal advice - provide general information instead
- Be supportive and understanding of housing concerns
- Use the retrieved documents to support your responses

Context from knowledge base:
{context}

Chat history:
{chat_history}

Current question: {question}

Provide a helpful, informative response about this housing-related question.
"""

TRANSLATION_TEMPLATE = """
You are a professional translator specializing in housing and legal terminology.

Task: Translate the following housing assistance response into {target_language}.

Guidelines:
- Maintain the professional and supportive tone
- Preserve all specific legal or housing terms accurately
- Keep the structure and formatting of the original response
- Ensure cultural appropriateness for the target language

Original response (in English):
{original_response}

Provide the translation in {target_language}:
"""

USER_TEMPLATE = "Question:```{question}```"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment variables
required_env_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise


# Singleton instances for resource reuse
_embeddings = None
_docsearch = None
_language_agent = None
_housing_agent = None
_translation_agent = None

def _get_embeddings() -> OpenAIEmbeddings:
    """Get or create embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings

def _get_docsearch() -> PineconeLangChain:
    """Get or create document search instance."""
    global _docsearch
    if _docsearch is None:
        try:
            # Try the standard LangChain approach first
            _docsearch = PineconeLangChain.from_existing_index(
                embedding=_get_embeddings(),
                index_name=INDEX_NAME,
            )
        except Exception as langchain_error:
            logger.warning(f"LangChain Pinecone wrapper failed: {langchain_error}")
            try:
                # Fallback: Create a simple wrapper around direct Pinecone client
                logger.info("Attempting direct Pinecone integration...")
                
                # This will require updating the HousingAssistantAgent to use direct Pinecone calls
                # For now, re-raise the original error
                logger.error(f"Failed to initialize document search: {langchain_error}")
                raise langchain_error
            except Exception as fallback_error:
                logger.error(f"Fallback Pinecone initialization also failed: {fallback_error}")
                raise
    return _docsearch

class LanguageDetectionAgent:
    """Agent responsible for detecting the language of user input."""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    def detect_language(self, text: str) -> Dict[str, Union[str, float]]:
        """Detect the language of the input text."""
        try:
            prompt = LANGUAGE_DETECTION_TEMPLATE.format(text=text)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse JSON response
            result = json.loads(response.content)
            logger.info(f"Detected language: {result}")
            return result
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            # Default to English if detection fails
            return {"language": "en", "confidence": 0.5}

class HousingAssistantAgent:
    """Agent responsible for providing housing-related assistance."""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=CHAT_TEMPERATURE, model="gpt-4")
        self.docsearch = None
        self.direct_pinecone = None
        self.embeddings = _get_embeddings()
        
        # Try to initialize document search
        try:
            self.docsearch = _get_docsearch()
            logger.info("Using LangChain Pinecone integration")
        except Exception as e:
            logger.warning(f"LangChain Pinecone failed, trying direct integration: {e}")
            try:
                # Initialize direct Pinecone client
                self.direct_pinecone = pc.Index(INDEX_NAME)
                logger.info("Using direct Pinecone integration")
            except Exception as direct_error:
                logger.error(f"Direct Pinecone also failed: {direct_error}")
                # Will work without knowledge base (LLM only)
    
    def _search_documents_direct(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search documents using direct Pinecone client."""
        try:
            # Get embedding for the question
            query_embedding = self.embeddings.embed_query(question)
            
            # Query Pinecone directly
            results = self.direct_pinecone.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                include_values=False
            )
            
            # Convert to LangChain-like format
            docs = []
            for match in results.get('matches', []):
                docs.append({
                    "metadata": match.get('metadata', {}),
                    "page_content": match.get('metadata', {}).get('text', 'No content available'),
                    "score": match.get('score', 0)
                })
            
            return docs
            
        except Exception as e:
            logger.error(f"Direct Pinecone search failed: {e}")
            return []
    
    def get_housing_assistance(self, question: str, chat_history: List[Dict[str, Any]], context: Any = None) -> Dict[str, Any]:
        """Provide housing assistance based on the question and retrieved context."""
        try:
            relevant_docs = []
            
            # Try to get relevant documents
            if self.docsearch:
                # Use LangChain integration
                try:
                    langchain_docs = self.docsearch.similarity_search(question, k=4)
                    relevant_docs = [{"metadata": doc.metadata, "page_content": doc.page_content} for doc in langchain_docs]
                except Exception as e:
                    logger.warning(f"LangChain search failed: {e}")
                    relevant_docs = []
            
            elif self.direct_pinecone:
                # Use direct Pinecone integration
                relevant_docs = self._search_documents_direct(question, k=4)
            
            # Extract context from documents
            if relevant_docs:
                context_str = "\n\n".join([doc.get("page_content", "") for doc in relevant_docs])
            else:
                context_str = "No specific documentation available. Please provide general housing guidance based on your knowledge."
                logger.warning("No document search available, using LLM knowledge only")
            
            # Format chat history
            history_str = ""
            if chat_history:
                history_str = "\n".join([
                    f"Human: {item.get('human', '')}\nAssistant: {item.get('system', '')}" 
                    for item in chat_history
                ])
            
            # Create prompt
            prompt = HOUSING_ASSISTANT_TEMPLATE.format(
                context=context_str,
                chat_history=history_str,
                question=question
            )
            
            # Get response
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            return {
                "answer": response.content,
                "source_documents": relevant_docs
            }
            
        except Exception as e:
            logger.error(f"Housing assistance failed: {e}")
            return {
                "answer": "I apologize, but I'm having trouble accessing my knowledge base right now. I can still provide general housing guidance based on my training. Please try again or ask a more specific question.",
                "source_documents": []
            }

class TranslationAgent:
    """Agent responsible for translating responses to the detected language."""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    def translate_response(self, response: str, target_language: str) -> str:
        """Translate the response to the target language."""
        # If target language is English, return as-is
        if target_language.lower() in ["en", "english"]:
            return response
        
        try:
            # Map common language codes to full names
            language_map = {
                "es": "Spanish", "fr": "French", "de": "German", 
                "nl": "Dutch", "it": "Italian", "pt": "Portuguese",
                "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
                "ko": "Korean", "ru": "Russian"
            }
            
            target_lang_name = language_map.get(target_language.lower(), target_language)
            
            prompt = TRANSLATION_TEMPLATE.format(
                target_language=target_lang_name,
                original_response=response
            )
            
            translated = self.llm.invoke([HumanMessage(content=prompt)])
            return translated.content
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return original response if translation fails
            return response

def _get_language_agent() -> LanguageDetectionAgent:
    """Get or create language detection agent."""
    global _language_agent
    if _language_agent is None:
        _language_agent = LanguageDetectionAgent()
    return _language_agent

def _get_housing_agent() -> HousingAssistantAgent:
    """Get or create housing assistant agent."""
    global _housing_agent
    if _housing_agent is None:
        _housing_agent = HousingAssistantAgent()
    return _housing_agent

def _get_translation_agent() -> TranslationAgent:
    """Get or create translation agent."""
    global _translation_agent
    if _translation_agent is None:
        _translation_agent = TranslationAgent()
    return _translation_agent

def _create_chat_prompt(use_custom_template: bool = False) -> ChatPromptTemplate:
    """Create chat prompt template."""
    if use_custom_template:
        system_template = """
### Instruction: You are a friendly customer support agent for a mobile application called **SIIP**, your name is 
     known as **Richard**. You help users navigate housing-related laws and regulations that may affect their housing rights, responsibilities, or situations.
     Use only the chat history and the following information:
     {context}
     to answer the question. If you do not know the answer – say you do not know.
     Keep your responses clear, informative, and supportive. Do not speculate or provide legal advice.
     {chat_history}
     ## Input: {question}
     ## Response:
        """
    else:
        system_template = SYSTEM_TEMPLATE
    
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(USER_TEMPLATE)
    ]
    return ChatPromptTemplate.from_messages(messages)

def _validate_inputs(query: str, context: Any, chat_history: List[Dict[str, Any]]) -> None:
    """Validate input parameters."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not isinstance(chat_history, list):
        raise TypeError("Chat history must be a list")
    
    for item in chat_history:
        if not isinstance(item, dict):
            raise TypeError("Chat history items must be dictionaries")

def _run_llm_with_agents(
    query: str, 
    chat_history: List[Dict[str, Any]], 
    context: Any = None,
    enable_translation: bool = True
) -> Dict[str, Any]:
    """
    Enhanced LLM runner using multi-agent system:
    1. Language Detection Agent - detects input language
    2. Housing Assistant Agent - provides housing expertise  
    3. Translation Agent - translates response to user's language
    """
    try:
        # Validate inputs
        _validate_inputs(query, context, chat_history)
        
        logger.info(f"Processing query with multi-agent system: {query[:50]}...")
        
        # Step 1: Detect language of user input
        language_agent = _get_language_agent()
        language_info = language_agent.detect_language(query)
        detected_language = language_info.get("language", "en")
        confidence = language_info.get("confidence", 0.5)
        
        logger.info(f"Detected language: {detected_language} (confidence: {confidence})")
        
        # Step 2: Get housing assistance (always in English for consistency with knowledge base)
        housing_agent = _get_housing_agent()
        housing_response = housing_agent.get_housing_assistance(query, chat_history, context)
        
        # Step 3: Translate response if needed
        final_answer = housing_response["answer"]
        
        if enable_translation and detected_language.lower() != "en" and confidence > 0.7:
            translation_agent = _get_translation_agent()
            final_answer = translation_agent.translate_response(
                housing_response["answer"], 
                detected_language
            )
            logger.info(f"Response translated to: {detected_language}")
        
        # Return enhanced result with language metadata
        result = {
            "answer": final_answer,
            "source_documents": housing_response["source_documents"],
            "language_info": {
                "detected_language": detected_language,
                "confidence": confidence,
                "translated": detected_language.lower() != "en" and confidence > 0.7
            }
        }
        
        logger.info("Multi-agent query processed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in multi-agent processing: {e}")
        # Fallback to simple response
        return {
            "answer": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
            "source_documents": [],
            "language_info": {
                "detected_language": "en",
                "confidence": 1.0,
                "translated": False
            }
        }

def _run_llm_base(
    context: Any, 
    query: str, 
    chat_history: List[Dict[str, Any]], 
    verbose: bool = DEFAULT_VERBOSE,
    use_custom_template: bool = False
) -> Dict[str, Any]:
    """Legacy LLM runner - maintained for backward compatibility."""
    # Use the new agent-based system
    return _run_llm_with_agents(query, chat_history, context)

def run_llm(context: dict, query: str, chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run multi-agent LLM system with language detection and translation.
    
    This function orchestrates three specialized agents:
    1. Language Detection Agent - identifies the user's input language
    2. Housing Assistant Agent - provides expert housing law assistance  
    3. Translation Agent - translates the response back to user's language
    
    Args:
        context: Dictionary containing context information (optional)
        query: User query string in any supported language
        chat_history: List of previous chat messages
        
    Returns:
        Dictionary containing:
        - answer: Response in the user's detected language
        - source_documents: Relevant documents from knowledge base
        - language_info: Metadata about language detection and translation
    """
    if chat_history is None:
        chat_history = []
    
    return _run_llm_with_agents(query, chat_history, context, enable_translation=True)

def run_llm_v1(context: str, query: str, chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Legacy interface - now uses the enhanced multi-agent system.
    
    Args:
        context: String containing context information
        query: User query string
        chat_history: List of previous chat messages
        
    Returns:
        Dictionary containing LLM response and source documents
    """
    if chat_history is None:
        chat_history = []
    
    return _run_llm_with_agents(query, chat_history, context, enable_translation=True)

def run_llm_english_only(context: Any, query: str, chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run LLM system without translation - English responses only.
    
    Useful for testing or when you specifically want English responses.
    
    Args:
        context: Context information (dict or string)
        query: User query string
        chat_history: List of previous chat messages
        
    Returns:
        Dictionary containing English response and source documents
    """
    if chat_history is None:
        chat_history = []
    
    return _run_llm_with_agents(query, chat_history, context, enable_translation=False)
