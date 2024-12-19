import streamlit as st
from typing import List, Dict

class ChatInterface:
    def __init__(self):
        # Inicializar el historial de chat en la sesión si no existe
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def display_message(self, message: Dict[str, str]):
        """Muestra un mensaje individual en el chat"""
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def display_chat_history(self):
        """Muestra todo el historial del chat"""
        for message in st.session_state.messages:
            self.display_message(message)

    def add_message(self, role: str, content: str):
        """Añade un nuevo mensaje al historial"""
        message = {"role": role, "content": content}
        st.session_state.messages.append(message)
        self.display_message(message)

    def get_user_input(self) -> str:
        """Obtiene la entrada del usuario"""
        return st.chat_input("Escribe tu mensaje aquí...")