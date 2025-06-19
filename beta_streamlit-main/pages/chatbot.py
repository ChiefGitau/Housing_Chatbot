import os
import streamlit as st
import time
from typing import Set
from loguru import logger
from streamlit_chat import message

# Import our local multi-agent system
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from core import run_llm, get_session_language

# Import global translation system
from global_translations import t, display_language_selector


def reset():
    if "chat_answers_history" in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if  "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

def history_valid(history):
    try:
        length = len(history)
        history_str = history[length - 1]
    except IndexError:
        history_str = 'No history'

    return history_str
def generate_response():

    for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
    ):
        message(
            user_query,
            is_user=True,
        )
        message(generated_response)
def update_hist(txt):
    try:
        index = len(st.session_state['chat_history']) - 1
        st.session_state['chat_history'][index]['system'] = txt
        st.session_state['chat_answers_history'][index] = txt
        st.write('history available')
    except IndexError:
        st.write('###no history available')


def hist_clicked(txt):
    update_hist(txt)
    st.write("Changes to conversation have been updated")

    message(
        txt,
        avatar_style= "pixel-art-neutral"
    )



def display_hist(history):

    try:
        length = len(history)
        history_str = history[length - 1]
    except IndexError:
        history_str = 'No histrory'
        history_str = 'No history available'


    return st.text_area(
        "Text to analyze",
        history_str,
        key = 'history_ed',
    )



def get_ai_response(question: str, chat_history: list, context: str = "") -> dict:
    """
    Get AI response using the local multi-agent system instead of external API.
    """
    try:
        logger.info(f"Processing question with multi-agent system: {question[:50]}...")
        logger.info(f"Chat history length: {len(chat_history)}")
        
        # Use our local multi-agent system
        result = run_llm(
            context={"additional_context": context} if context else {},
            query=question,
            chat_history=chat_history
        )
        
        logger.info("Multi-agent system response received successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        # Return error response in expected format
        return {
            "answer": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
            "source_documents": [],
            "language_info": {
                "detected_language": "en",
                "confidence": 1.0,
                "translated": False
            }
        }


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

def display_convo(user, bot):

    # if st.chat_message('assistant'):
    #     logger.info(f"the fucker was hiding here: {st.chat_message}")

    if user is None:
        return

    with st.chat_message("assistant"):

        # st.markdown(response)
        # Add assistant response to chat history
        for generated_response, user_query in zip(
                bot,
                user,
        ):
            message(
                user_query,
                is_user=True,
                # key='assistant'
            )
            message(generated_response)


def laisa(prompt):
    """Process user input with multi-agent system and update chat history."""
    
    logger.info(f"Processing user prompt: {prompt[:50]}...")
    
    # Store current language preference before processing
    current_lang_pref = st.session_state.get('user_language_preference', 'en')
    
    # Get AI response using local multi-agent system
    generated_response = get_ai_response(
        question=prompt,
        chat_history=st.session_state['chat_history'],
        context=""
    )

    logger.info(f"Generated response received")
    
    # Extract sources for display
    try:
        sources = set()
        if 'source_documents' in generated_response:
            for doc in generated_response["source_documents"]:
                if isinstance(doc, dict) and 'metadata' in doc:
                    if 'source' in doc['metadata']:
                        sources.add(doc['metadata']['source'])
                    elif 'source_file' in doc['metadata']:
                        sources.add(doc['metadata']['source_file'])
    except (KeyError, TypeError) as e:
        logger.warning(f"Error extracting sources: {e}")
        sources = set()

    # Format response with sources
    answer = generated_response.get('answer', 'No response available')
    sources_string = create_sources_string(sources)
    formatted_response = f"{answer}"
    if sources_string:
        formatted_response += f"\n\n{sources_string}"
    
    # Add language info to response if available
    language_changed = False
    if 'language_info' in generated_response:
        lang_info = generated_response['language_info']
        # Store language info for sidebar display
        st.session_state.last_language_info = lang_info
        if lang_info.get('translated', False):
            formatted_response += f"\n\n*Translated from {lang_info.get('detected_language', 'unknown')} ({t('confidence')}: {lang_info.get('confidence', 0):.1%})*"
        
        # Check if language preference was automatically updated
        new_lang_pref = st.session_state.get('user_language_preference', 'en')
        if new_lang_pref != current_lang_pref:
            language_changed = True
            logger.info(f"Language preference changed from {current_lang_pref} to {new_lang_pref}")

    # Update session state
    st.session_state.chat_history.append({
        'human': prompt, 
        'system': answer
    })
    st.session_state.user_prompt_history.append(prompt)
    st.session_state.chat_answers_history.append(formatted_response)

    logger.info(f"Chat history updated. Total messages: {len(st.session_state.chat_history)}")
    
    # If language was automatically detected and changed, trigger a rerun to update all page text
    if language_changed:
        logger.info("Triggering page refresh due to language change")
        st.rerun()


st.header(f"LAISA - {t('housing_assistant')} 🏠🤖")
st.markdown(f"*{t('multilingual_ai_assistant')}*")

# Display language selector in sidebar
with st.sidebar:
    display_language_selector()

# Initialize chat history
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    st.session_state.key= 'main_session'

update = False



if prompt := st.chat_input("Ask me about housing laws, tenant rights, or regulations... (any language)"):
    laisa(prompt)
    update = True




col1, col2, col3 = st.columns(3)

with col2:
    if st.button(f'{t("reset_chat")} :repeat:', use_container_width=True):
        reset()
        st.rerun()


with st.sidebar:
    st.markdown(f"### 🌐 {t('language_support')}")
    st.markdown("""
    LAISA automatically detects your language and responds accordingly:
    - 🇺🇸 English
    - 🇪🇸 Spanish  
    - 🇫🇷 French
    - 🇩🇪 German
    - 🇳🇱 Dutch
    - 🇮🇹 Italian
    - 🇵🇹 Portuguese
    - 🇸🇦 Arabic
    - 🇨🇳 Chinese
    - 🇯🇵 Japanese
    - 🇰🇷 Korean
    - 🇷🇺 Russian
    """)
    
    st.markdown(f"### 🏠 {t('topics_help_with')}")
    st.markdown(f"""
    - {t('tenant_rights')}
    - {t('landlord_obligations')}
    - {t('rental_agreements')}
    - {t('housing_regulations')}
    - {t('eviction_processes')}
    - {t('property_management')}
    - {t('housing_discrimination')}
    """)
    
    if st.session_state.get("chat_answers_history"):
        st.markdown(f"### 📝 {t('current_session')}")
        st.info(f"{t('messages')}: {len(st.session_state['chat_answers_history'])}")
        
        # Show language consistency info
        try:
            session_lang = get_session_language()
            if session_lang.get('language') != 'en':
                lang_code = session_lang.get('language', 'unknown').upper()
                confidence = session_lang.get('confidence', 0)
                source = session_lang.get('source', 'unknown')
                # Indicate if confidence meets threshold
                threshold_indicator = "✅" if confidence >= 0.7 else "⚠️"
                st.success(f"🌐 {t('active_language')}: {lang_code} ({confidence:.1%}) {threshold_indicator}")
                st.caption(f"{t('source')}: {source} | {t('translation_active') if confidence >= 0.7 else t('translation_inactive')}")
                
                # Show enhanced language info if available
                if hasattr(st.session_state, 'last_language_info'):
                    lang_info = st.session_state.last_language_info
                    script = lang_info.get('script', 'Unknown')
                    indicators = lang_info.get('indicators', [])
                    alternatives = lang_info.get('alternatives', [])
                    detection_method = lang_info.get('detection_method', 'unknown')
                    
                    if script != 'Unknown':
                        st.caption(f"📝 Script: {script}")
                    
                    # Show detection method
                    st.caption(f"⚙️ Method: {detection_method.replace('_', ' ').title()}")
                    
                    if indicators:
                        st.caption(f"🔍 Indicators: {', '.join(indicators[:2])}")
                    
                    # Show statistical vs AI confidence if available
                    analysis = lang_info.get('analysis', {})
                    if analysis.get('statistical_confidence') or analysis.get('ai_confidence'):
                        stat_conf = analysis.get('statistical_confidence', 0)
                        ai_conf = analysis.get('ai_confidence', 0)
                        if stat_conf > 0:
                            st.caption(f"📊 Statistical: {stat_conf:.1%}")
                        if ai_conf > 0:
                            st.caption(f"🤖 AI: {ai_conf:.1%}")
                    
                    # Show alternative language candidates
                    if alternatives:
                        st.caption("**Alternative candidates:**")
                        for i, alt in enumerate(alternatives[:2]):  # Show top 2 alternatives
                            alt_lang = alt.get('language', 'unknown').upper()
                            alt_conf = alt.get('confidence', 0)
                            reasoning = alt.get('reasoning', 'unknown')
                            st.caption(f"  {i+2}. {alt_lang}: {alt_conf:.1%} - {reasoning[:25]}{'...' if len(reasoning) > 25 else ''}")
            else:
                st.info("🇺🇸 Language: English (Default)")
        except Exception as e:
            # Fallback to original language info display
            if hasattr(st.session_state, 'last_language_info'):
                lang_info = st.session_state.last_language_info
                if lang_info.get('detected_language') != 'en':
                    st.success(f"🌐 Detected: {lang_info.get('detected_language', 'unknown').upper()}")
    
    # txt = display_hist(st.session_state["chat_answers_history"])
    # st.button(':white_check_mark:', on_click=hist_clicked(txt), key='change_hist')



display_convo(st.session_state['user_prompt_history'], st.session_state['chat_answers_history'])



