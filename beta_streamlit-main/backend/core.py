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

# Language detection libraries
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

import prompt_template as prompt_template

load_dotenv()

# Configuration constants
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "small-blogs-emmbeddings-index"
CHAT_TEMPERATURE = 0
DEFAULT_VERBOSE = False

# Agent Templates
LANGUAGE_DETECTION_TEMPLATE = """
You are an expert multilingual language detection specialist. Analyze the user input with extreme precision using multiple detection methods.

Analyze the following text considering:
1. Writing system/alphabet (Latin, Cyrillic, Arabic, CJK ideographs, Hiragana, Katakana, Hangul)
2. Language-specific patterns (grammar structures, word order, morphology)
3. Vocabulary and common phrases (function words, prepositions, articles)
4. Orthographic features (diacritics, capitalization, punctuation patterns)
5. Character frequency and n-gram patterns
6. Language-specific linguistic markers

Provide the TOP 3 most likely languages with their confidence scores.

Return ONLY a JSON response:
{
    "primary": {
        "language": "language_code",
        "confidence": confidence_score,
        "script": "writing_system",
        "indicators": ["specific_linguistic_features"]
    },
    "alternatives": [
        {
            "language": "second_most_likely",
            "confidence": confidence_score,
            "reasoning": "why_this_language_considered"
        },
        {
            "language": "third_most_likely", 
            "confidence": confidence_score,
            "reasoning": "why_this_language_considered"
        }
    ],
    "analysis": {
        "text_length": character_count,
        "script_dominant": "primary_script",
        "linguistic_complexity": "simple|moderate|complex"
    }
}

Where:
- language_code is ISO 639-1 code (en, es, fr, de, nl, it, pt, ar, zh, ja, ko, ru, etc.)
- confidence_score is between 0 and 1 (be precise and conservative)
- script is the writing system identified
- indicators are specific linguistic features that led to detection
- Ensure alternative languages have meaningful confidence differences

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
_language_consistency_agent = None
_ui_translation_agent = None

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
    """Agent responsible for detecting the language of user input using advanced script and content analysis."""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")  # Using GPT-4 for better script analysis
        # Set langdetect to deterministic mode for consistent results
        DetectorFactory.seed = 0
    
    def _analyze_script(self, text: str) -> Dict[str, str]:
        """Quick script analysis using character ranges."""
        import unicodedata
        
        script_indicators = {
            'Latin': 0, 'Cyrillic': 0, 'Arabic': 0, 'Han': 0, 
            'Hiragana': 0, 'Katakana': 0, 'Hangul': 0
        }
        
        for char in text:
            if char.isalpha():
                try:
                    script_name = unicodedata.name(char).split()[0]
                    if 'LATIN' in script_name:
                        script_indicators['Latin'] += 1
                    elif 'CYRILLIC' in script_name:
                        script_indicators['Cyrillic'] += 1
                    elif 'ARABIC' in script_name:
                        script_indicators['Arabic'] += 1
                    elif 'CJK' in script_name or 'IDEOGRAPH' in script_name:
                        script_indicators['Han'] += 1
                    elif 'HIRAGANA' in script_name:
                        script_indicators['Hiragana'] += 1
                    elif 'KATAKANA' in script_name:
                        script_indicators['Katakana'] += 1
                    elif 'HANGUL' in script_name:
                        script_indicators['Hangul'] += 1
                except ValueError:
                    continue
        
        # Find dominant script
        dominant_script = max(script_indicators, key=script_indicators.get)
        total_chars = sum(script_indicators.values())
        
        if total_chars == 0:
            return {"script": "Unknown", "confidence": 0.0}
        
        script_confidence = script_indicators[dominant_script] / total_chars
        return {"script": dominant_script, "confidence": script_confidence}
    
    def _statistical_language_detection(self, text: str) -> Dict[str, Union[str, float]]:
        """Fast statistical language detection using langdetect library."""
        try:
            # Clean text for better detection
            clean_text = text.strip()
            
            # Require minimum length for reliable detection
            if len(clean_text) < 3:
                return {"language": "unknown", "confidence": 0.0, "method": "text_too_short"}
            
            # Use langdetect for statistical analysis
            detected_lang = detect(clean_text)
            
            # Get detailed probabilities
            from langdetect import detect_langs
            lang_probs = detect_langs(clean_text)
            
            # Find confidence for detected language
            primary_confidence = 0.0
            alternatives = []
            
            for lang_prob in lang_probs:
                if lang_prob.lang == detected_lang:
                    primary_confidence = lang_prob.prob
                else:
                    alternatives.append({
                        "language": lang_prob.lang,
                        "confidence": lang_prob.prob
                    })
            
            # Sort alternatives by confidence
            alternatives.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"Statistical detection: {detected_lang} ({primary_confidence:.3f})")
            
            return {
                "language": detected_lang,
                "confidence": primary_confidence,
                "alternatives": alternatives[:2],  # Top 2 alternatives
                "method": "statistical"
            }
            
        except LangDetectException as e:
            logger.warning(f"Langdetect failed: {e}")
            return {"language": "unknown", "confidence": 0.0, "method": "detection_failed"}
        except Exception as e:
            logger.error(f"Statistical detection error: {e}")
            return {"language": "unknown", "confidence": 0.0, "method": "error"}
    
    def detect_language(self, text: str) -> Dict[str, Union[str, float, List[str]]]:
        """Detect the language of the input text using hybrid analysis (statistical + AI + script)."""
        try:
            # Step 1: Fast statistical detection
            statistical_result = self._statistical_language_detection(text)
            
            # Step 2: Script analysis for additional context
            script_analysis = self._analyze_script(text)
            
            # Step 3: AI-based linguistic analysis (for complex cases or validation)
            ai_result = None
            use_ai_analysis = (
                statistical_result["confidence"] < 0.85 or  # Low statistical confidence
                statistical_result["language"] == "unknown" or  # Failed detection
                len(text) > 100  # Long text that benefits from linguistic analysis
            )
            
            if use_ai_analysis:
                try:
                    prompt = LANGUAGE_DETECTION_TEMPLATE.format(text=text)
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    
                    # Parse JSON response
                    try:
                        ai_result = json.loads(response.content)
                    except json.JSONDecodeError:
                        # Try to extract JSON from response if it's embedded in text
                        import re
                        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                        if json_match:
                            ai_result = json.loads(json_match.group())
                except Exception as ai_error:
                    logger.warning(f"AI analysis failed: {ai_error}")
                    ai_result = None
            
            # Step 4: Combine results intelligently
            final_result = self._combine_detection_results(
                statistical_result, script_analysis, ai_result, text
            )
            
            logger.info(f"Hybrid detection: {final_result['language']}({final_result['confidence']:.2f}) via {final_result.get('detection_method', 'unknown')}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Hybrid language detection failed: {e}")
            # Ultimate fallback with script analysis
            script_analysis = self._analyze_script(text)
            fallback_lang = self._script_to_language_fallback(script_analysis["script"])
            
            return {
                "language": fallback_lang,
                "confidence": max(0.3, script_analysis["confidence"] * 0.6),
                "script": script_analysis["script"],
                "indicators": ["fallback_script_analysis"],
                "alternatives": [],
                "analysis": {"text_length": len(text), "script_dominant": script_analysis["script"]},
                "script_analysis": script_analysis,
                "detection_method": "fallback"
            }
    
    def _combine_detection_results(self, statistical_result, script_analysis, ai_result, text):
        """Intelligently combine results from different detection methods."""
        # Start with statistical result as base
        primary_language = statistical_result.get("language", "en")
        primary_confidence = statistical_result.get("confidence", 0.0)
        detection_method = "statistical"
        
        # If AI analysis is available and either confirms or contradicts statistical result
        if ai_result:
            ai_primary = ai_result.get("primary", {})
            ai_language = ai_primary.get("language", "en")
            ai_confidence = ai_primary.get("confidence", 0.0)
            
            # If AI and statistical agree, boost confidence
            if ai_language == primary_language:
                combined_confidence = min(1.0, (primary_confidence + ai_confidence) / 2 + 0.1)
                detection_method = "statistical+ai_confirmed"
                logger.info(f"Statistical and AI agree on {primary_language}, confidence boosted")
            
            # If AI has much higher confidence, use AI result
            elif ai_confidence > primary_confidence + 0.2:
                primary_language = ai_language
                primary_confidence = ai_confidence
                detection_method = "ai_override"
                logger.info(f"AI override: {ai_language} ({ai_confidence:.2f}) > statistical ({primary_confidence:.2f})")
            
            # If statistical has good confidence, stick with it but note disagreement
            elif primary_confidence > 0.8:
                detection_method = "statistical_high_confidence"
                logger.info(f"Statistical confident: {primary_language} vs AI: {ai_language}")
            
            # If both have low confidence, choose the higher one
            else:
                if ai_confidence > primary_confidence:
                    primary_language = ai_language
                    primary_confidence = ai_confidence
                    detection_method = "ai_better_confidence"
                else:
                    detection_method = "statistical_better_confidence"
        
        # Script validation and confidence adjustment
        if script_analysis["confidence"] > 0.8:
            if self._script_language_alignment(primary_language, script_analysis["script"]):
                primary_confidence = min(1.0, primary_confidence + 0.05)
                detection_method += "+script_confirmed"
            else:
                # Script doesn't align - reduce confidence
                primary_confidence = max(0.3, primary_confidence - 0.1)
                detection_method += "+script_conflict"
        
        # Compile alternatives from all sources
        alternatives = []
        
        # Add statistical alternatives
        if statistical_result.get("alternatives"):
            for alt in statistical_result["alternatives"][:2]:
                alternatives.append({
                    "language": alt["language"],
                    "confidence": alt["confidence"],
                    "reasoning": "statistical_analysis"
                })
        
        # Add AI alternatives if available
        if ai_result and ai_result.get("alternatives"):
            for alt in ai_result["alternatives"][:2]:
                # Avoid duplicates
                existing_langs = [a["language"] for a in alternatives]
                if alt.get("language") not in existing_langs:
                    alternatives.append({
                        "language": alt.get("language"),
                        "confidence": alt.get("confidence", 0),
                        "reasoning": alt.get("reasoning", "ai_analysis")
                    })
        
        # Sort alternatives by confidence
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "language": primary_language,
            "confidence": primary_confidence,
            "script": script_analysis["script"],
            "indicators": statistical_result.get("indicators", []) + (ai_result.get("primary", {}).get("indicators", []) if ai_result else []),
            "alternatives": alternatives[:3],  # Top 3 alternatives
            "analysis": {
                "text_length": len(text),
                "script_dominant": script_analysis["script"],
                "statistical_confidence": statistical_result.get("confidence", 0),
                "ai_confidence": ai_result.get("primary", {}).get("confidence", 0) if ai_result else 0
            },
            "script_analysis": script_analysis,
            "detection_method": detection_method
        }
    
    def _script_language_alignment(self, language: str, script: str) -> bool:
        """Check if detected language aligns with script analysis."""
        alignments = {
            'en': ['Latin'], 'es': ['Latin'], 'fr': ['Latin'], 'de': ['Latin'], 
            'nl': ['Latin'], 'it': ['Latin'], 'pt': ['Latin'],
            'ru': ['Cyrillic'], 'ar': ['Arabic'], 'zh': ['Han'],
            'ja': ['Hiragana', 'Katakana', 'Han'], 'ko': ['Hangul']
        }
        expected_scripts = alignments.get(language, [])
        return script in expected_scripts
    
    def _script_to_language_fallback(self, script: str) -> str:
        """Fallback language detection based on script."""
        script_defaults = {
            'Cyrillic': 'ru', 'Arabic': 'ar', 'Han': 'zh',
            'Hiragana': 'ja', 'Katakana': 'ja', 'Hangul': 'ko',
            'Latin': 'en'  # Default to English for Latin script
        }
        return script_defaults.get(script, 'en')

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

class LanguageConsistencyAgent:
    """Agent responsible for ensuring language consistency across the application."""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.language_map = {
            "es": "Spanish", "fr": "French", "de": "German", 
            "nl": "Dutch", "it": "Italian", "pt": "Portuguese",
            "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
            "ko": "Korean", "ru": "Russian"
        }
    
    def get_session_language(self) -> Dict[str, Union[str, float]]:
        """Get the consistent language for the session."""
        try:
            # First check if there's a stored session language
            import streamlit as st
            if hasattr(st.session_state, 'user_language_preference'):
                lang_code = st.session_state.user_language_preference
                return {
                    "language": lang_code,
                    "confidence": 1.0,
                    "source": "session_preference"
                }
            
            # Check if language was detected in chatbot
            if hasattr(st.session_state, 'last_language_info'):
                lang_info = st.session_state.last_language_info
                if lang_info.get('confidence', 0) >= 0.7:
                    return {
                        "language": lang_info.get('detected_language', 'en'),
                        "confidence": lang_info.get('confidence', 0.8),
                        "source": "chatbot_detection"
                    }
        except:
            pass
        
        # Default to English
        return {
            "language": "en",
            "confidence": 1.0,
            "source": "default"
        }
    
    def set_session_language(self, language_code: str, confidence: float = 1.0):
        """Set the session language preference."""
        try:
            import streamlit as st
            st.session_state.user_language_preference = language_code
            st.session_state.last_language_info = {
                "detected_language": language_code,
                "confidence": confidence,
                "translated": language_code.lower() != "en"
            }
        except:
            pass
    
    def ensure_response_language(self, response: str, force_language: str = None) -> str:
        """Ensure response is in the correct language for the session."""
        target_lang_info = self.get_session_language()
        target_language = force_language or target_lang_info["language"]
        
        # If target is English, return as-is
        if target_language.lower() in ["en", "english"]:
            return response
        
        # If confidence is low, don't translate
        if target_lang_info["confidence"] < 0.7 and not force_language:
            return response
        
        try:
            target_lang_name = self.language_map.get(target_language.lower(), target_language)
            
            prompt = TRANSLATION_TEMPLATE.format(
                target_language=target_lang_name,
                original_response=response
            )
            
            translated = self.llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"Response translated to {target_language} (confidence: {target_lang_info['confidence']})")
            return translated.content
            
        except Exception as e:
            logger.error(f"Language consistency translation failed: {e}")
            return response


class UITranslationAgent:
    """Agent for dynamically translating user interface text to any detected language."""
    
    def __init__(self):
        """Initialize the UI translation agent."""
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.translation_cache = {}  # Cache translations to avoid repeated API calls
        
        # Template for UI text translation
        self.ui_translation_template = """You are a professional UI translator. Translate the following user interface text to {target_language}.

Rules:
1. Keep the translation concise and appropriate for UI elements
2. Maintain the original meaning and context
3. Use standard UI terminology for the target language
4. For technical terms, use commonly accepted translations
5. Return ONLY the translated text, no explanations

Original text: "{original_text}"

Translation:"""
    
    def translate_ui_text(self, text: str, target_language: str) -> str:
        """
        Translate UI text to the target language.
        
        Args:
            text: The English text to translate
            target_language: Target language code (e.g., 'es', 'fr', 'ar')
            
        Returns:
            Translated text
        """
        # If target is English, return as-is
        if target_language.lower() in ["en", "english"]:
            return text
        
        # Check cache first
        cache_key = f"{text}_{target_language}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Language mapping for better context
            language_map = {
                "es": "Spanish", "fr": "French", "de": "German", 
                "nl": "Dutch", "it": "Italian", "pt": "Portuguese",
                "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
                "ko": "Korean", "ru": "Russian", "tr": "Turkish",
                "pl": "Polish", "sv": "Swedish", "da": "Danish",
                "no": "Norwegian", "fi": "Finnish", "cs": "Czech",
                "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian",
                "hr": "Croatian", "sk": "Slovak", "sl": "Slovenian"
            }
            
            target_lang_name = language_map.get(target_language.lower(), target_language.capitalize())
            
            prompt = self.ui_translation_template.format(
                target_language=target_lang_name,
                original_text=text
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            translated_text = response.content.strip()
            
            # Cache the translation
            self.translation_cache[cache_key] = translated_text
            
            logger.info(f"UI text translated to {target_language}: '{text}' -> '{translated_text}'")
            return translated_text
            
        except Exception as e:
            logger.error(f"UI translation failed for '{text}' to {target_language}: {e}")
            return text  # Return original text if translation fails
    
    def translate_multiple(self, texts: list, target_language: str) -> dict:
        """
        Translate multiple UI texts efficiently.
        
        Args:
            texts: List of English texts to translate
            target_language: Target language code
            
        Returns:
            Dictionary mapping original text to translated text
        """
        if target_language.lower() in ["en", "english"]:
            return {text: text for text in texts}
        
        translations = {}
        uncached_texts = []
        
        # Check cache for existing translations
        for text in texts:
            cache_key = f"{text}_{target_language}"
            if cache_key in self.translation_cache:
                translations[text] = self.translation_cache[cache_key]
            else:
                uncached_texts.append(text)
        
        # Translate uncached texts in batch
        if uncached_texts:
            try:
                language_map = {
                    "es": "Spanish", "fr": "French", "de": "German", 
                    "nl": "Dutch", "it": "Italian", "pt": "Portuguese",
                    "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
                    "ko": "Korean", "ru": "Russian", "tr": "Turkish",
                    "pl": "Polish", "sv": "Swedish", "da": "Danish",
                    "no": "Norwegian", "fi": "Finnish", "cs": "Czech",
                    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian",
                    "hr": "Croatian", "sk": "Slovak", "sl": "Slovenian"
                }
                
                target_lang_name = language_map.get(target_language.lower(), target_language.capitalize())
                
                # Create batch translation prompt
                batch_prompt = f"""You are a professional UI translator. Translate the following user interface texts to {target_lang_name}.

Rules:
1. Keep translations concise and appropriate for UI elements
2. Maintain original meaning and context
3. Use standard UI terminology for the target language
4. Return translations in the same order, one per line
5. Return ONLY the translations, no explanations

Texts to translate:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(uncached_texts)])}

Translations:"""

                response = self.llm.invoke([HumanMessage(content=batch_prompt)])
                translated_lines = response.content.strip().split('\n')
                
                # Map translations back to original texts
                for i, text in enumerate(uncached_texts):
                    if i < len(translated_lines):
                        translated = translated_lines[i].strip()
                        # Remove numbering if present
                        if translated.startswith(f"{i+1}."):
                            translated = translated[len(f"{i+1}."):].strip()
                        
                        translations[text] = translated
                        # Cache the translation
                        cache_key = f"{text}_{target_language}"
                        self.translation_cache[cache_key] = translated
                    else:
                        translations[text] = text  # Fallback to original
                
                logger.info(f"Batch translated {len(uncached_texts)} UI texts to {target_language}")
                
            except Exception as e:
                logger.error(f"Batch UI translation failed to {target_language}: {e}")
                # Fallback to individual translations
                for text in uncached_texts:
                    translations[text] = self.translate_ui_text(text, target_language)
        
        return translations


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

def _get_language_consistency_agent() -> LanguageConsistencyAgent:
    """Get or create language consistency agent."""
    global _language_consistency_agent
    if _language_consistency_agent is None:
        _language_consistency_agent = LanguageConsistencyAgent()
    return _language_consistency_agent

def _get_ui_translation_agent() -> UITranslationAgent:
    """Get or create UI translation agent."""
    global _ui_translation_agent
    if _ui_translation_agent is None:
        _ui_translation_agent = UITranslationAgent()
    return _ui_translation_agent

def get_session_language() -> Dict[str, Union[str, float]]:
    """Public function to get the current session language."""
    agent = _get_language_consistency_agent()
    return agent.get_session_language()

def ensure_response_language(response: str, force_language: str = None) -> str:
    """Public function to ensure response is in correct language."""
    agent = _get_language_consistency_agent()
    return agent.ensure_response_language(response, force_language)

def translate_ui_text(text: str, target_language: str = None) -> str:
    """Public function to translate UI text dynamically."""
    if target_language is None:
        # Auto-detect target language from session
        session_lang = get_session_language()
        target_language = session_lang.get("language", "en")
        
        # Only translate if confidence is high enough
        if session_lang.get("confidence", 0) < 0.7:
            return text
    
    agent = _get_ui_translation_agent()
    return agent.translate_ui_text(text, target_language)

def translate_ui_texts(texts: list, target_language: str = None) -> dict:
    """Public function to translate multiple UI texts efficiently."""
    if target_language is None:
        # Auto-detect target language from session
        session_lang = get_session_language()
        target_language = session_lang.get("language", "en")
        
        # Only translate if confidence is high enough
        if session_lang.get("confidence", 0) < 0.7:
            return {text: text for text in texts}
    
    agent = _get_ui_translation_agent()
    return agent.translate_multiple(texts, target_language)

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
        
        # Step 1.5: Automatically update global language preference if confidence is high
        if confidence >= 0.7 and detected_language != "en":
            try:
                # Import global translation functions
                from global_translations import set_language_preference
                
                # Automatically set the detected language as global preference
                if set_language_preference(detected_language):
                    logger.info(f"Global language preference automatically updated to: {detected_language}")
                else:
                    logger.warning(f"Failed to set global language preference to: {detected_language}")
            except ImportError:
                logger.warning("Global translation system not available - skipping global language update")
            except Exception as e:
                logger.error(f"Error updating global language preference: {e}")
        
        # Step 2: Get housing assistance (always in English for consistency with knowledge base)
        housing_agent = _get_housing_agent()
        housing_response = housing_agent.get_housing_assistance(query, chat_history, context)
        
        # Step 3: Translate response if needed
        final_answer = housing_response["answer"]
        
        if enable_translation and detected_language.lower() != "en" and confidence >= 0.7:
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
                "translated": detected_language.lower() != "en" and confidence > 0.7,
                # Include additional detection metadata
                "script": language_info.get("script", "Unknown"),
                "indicators": language_info.get("indicators", []),
                "detection_method": language_info.get("detection_method", "hybrid"),
                "analysis": language_info.get("analysis", {}),
                "alternatives": language_info.get("alternatives", []),
                "source": "chatbot_detection"
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
