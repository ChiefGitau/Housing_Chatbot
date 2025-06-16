import os
import streamlit as st
from typing import Set
from streamlit_chat import message
from backend.core import run_llm_v1
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string



st.header("Housing Help🦜🔗 ")


# Initialize chat history
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    st.session_state.key = "A"


if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history


    # Get context from environment or use default housing context
    context = os.getenv('LAISA_CONTEXT', 
                       'You are a helpful assistant specializing in housing-related laws and regulations.')
    
    generated_response = run_llm_v1(
        context=context, query=prompt, chat_history=st.session_state["chat_history"]
    )


    sources = set(
        [doc.metadata["source"] for doc in generated_response["source_documents"]]
    )
    formatted_response = (
        f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
    )

    # response = f"Echo: {formatted_response}"


    st.session_state.chat_history.append((prompt, generated_response["answer"]))
    st.session_state.user_prompt_history.append(prompt)
    st.session_state.chat_answers_history.append(formatted_response)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # st.markdown(response)
    # Add assistant response to chat history
        for generated_response, user_query in zip(
                st.session_state["chat_answers_history"],
                st.session_state["user_prompt_history"],
            ):

                message(
                    user_query,
                    is_user=True,
                )
                message(generated_response)
    # st.session_state.messages.append({"role": "assistant", "content": response})


