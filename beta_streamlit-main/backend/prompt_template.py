from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import Dict, List, Any, Optional


class PromptTemplate:
    def __init__(self, history: Optional[List[Dict[str, Any]]] = None, context: Optional[Dict[str, Any]] = None):
        self.history = history if history else []
        self.context = context if context else {}

    def create_system_message_prompt(self, system_template: str) -> str:
        """Create system message prompt with history and context."""
        history_str = "\n".join([f"System:{item['system']} \n Human:{item['human']}" for item in self.history])
        context_str = "\n".join([f"{key}: {value}" for key, value in self.context.items()])
        
        try:
            system_prompt = system_template.format(
                chat_history=history_str, 
                data=context_str, 
                context=context_str
            )
        except KeyError as e:
            # Fallback if template formatting fails
            system_prompt = system_template.replace("{chat_history}", history_str).replace("{data}", context_str).replace("{context}", context_str)
        
        return system_prompt



    def create_human_message_prompt(self, human_template: str, human_message: str) -> str:
        """Create human message prompt with the user's question."""
        try:
            human_prompt = human_template.format(question=human_message)
        except KeyError:
            # Fallback if template formatting fails
            human_prompt = human_template.replace("{question}", human_message)
        
        return human_prompt

    def create_chat_prompt(self, system_template: str, human_template: str, human_message: str) -> ChatPromptTemplate:
        """Create complete chat prompt with system and human messages."""
        system_prompt = self.create_system_message_prompt(system_template)
        human_prompt = self.create_human_message_prompt(human_template, human_message)

        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ]
        return ChatPromptTemplate.from_messages(messages)