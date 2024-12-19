import streamlit as st
from frontend.components.chat_interface import ChatInterface

class ChatPage:
    def __init__(self):
        self.setup_page()
        self.chat_interface = ChatInterface()
        self.setup_sidebar()
        self.show_welcome_message()

    def setup_page(self):
        st.set_page_config(
            page_title="Asistente Legal",
            page_icon="⚖️",
            layout="centered"
        )
        st.header("🤖 Asistente Legal")

    def setup_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Configuración")
            
            # Selector de temperatura (afectará a la creatividad de las respuestas)
            st.slider(
                "Temperatura",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controla la creatividad de las respuestas"
            )
            
            # Botón para limpiar la conversación
            if st.button("🗑️ Limpiar conversación"):
                st.session_state.messages = []
                st.rerun()

            # Ejemplos de preguntas
            st.header("📝 Ejemplos de preguntas")
            ejemplos = [
                "¿Cuál es el proceso para presentar una demanda civil?",
                "¿Qué documentos necesito para crear una sociedad limitada?",
                "Explícame los derechos básicos de un trabajador"
            ]
            
            for ejemplo in ejemplos:
                if st.button(ejemplo):
                    self.chat_interface.add_message("user", ejemplo)
                    response = self.get_model_response(ejemplo)
                    self.chat_interface.add_message("assistant", response)
                    st.rerun()

    def show_welcome_message(self):
        if "welcome_shown" not in st.session_state:
            welcome_msg = {
                "role": "assistant",
                "content": """¡Hola! Soy tu Asistente Legal. 👋

Estoy aquí para ayudarte con consultas legales básicas. Puedes preguntarme sobre:
- Procesos legales
- Documentación necesaria
- Derechos y obligaciones
- Interpretación de leyes
                
¿En qué puedo ayudarte hoy?"""
            }
            self.chat_interface.add_message(welcome_msg["role"], welcome_msg["content"])
            st.session_state.welcome_shown = True

    def get_model_response(self, user_input: str) -> str:
        """Simulación temporal de respuesta"""
        return f"Esta sería la respuesta legal a tu pregunta sobre: {user_input}\n\nPor ahora es solo una simulación, pronto conectaremos con el modelo real."

    def run(self):
        self.chat_interface.display_chat_history()
        user_input = self.chat_interface.get_user_input()

        if user_input:
            self.chat_interface.add_message("user", user_input)
            with st.spinner('Analizando tu consulta...'):
                response = self.get_model_response(user_input)
                self.chat_interface.add_message("assistant", response)