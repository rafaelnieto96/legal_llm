import streamlit as st
from frontend.components.chat_interface import ChatInterface
from models.model import LegalLLM
from tokenizer.tokenizer import SimpleTokenizer
import torch

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
        
        # Aplicar tema oscuro
        st.markdown("""
            <style>
                [data-testid="stAppViewContainer"] {
                    background-color: #0E1117;
                }
                .stMarkdown {
                    color: white;
                }
                .stButton button {
                    background-color: #262730;
                    color: white;
                }
                .stSlider {
                    color: white;
                }
                [data-testid="stSidebar"] {
                    background-color: #262730;
                }
                .stHeader {
                    color: white;
                }
                [data-testid="stChatMessage"] {
                    background-color: #262730;
                    color: white;
                }
                [data-testid="stMarkdownContainer"] {
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
        
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
        """Usa el modelo entrenado para generar respuestas"""
        # Cargar el modelo y tokenizador
        checkpoint = torch.load('legal_llm_model.pth')
        
        # Inicializar tokenizador con el vocabulario guardado
        tokenizer = SimpleTokenizer()
        tokenizer.word_to_idx = checkpoint['tokenizer_vocab']
        tokenizer.vocab_size = len(tokenizer.word_to_idx)
        tokenizer.idx_to_word = {v: k for k, v in tokenizer.word_to_idx.items()}
        
        # Inicializar y cargar modelo
        model = LegalLLM(tokenizer.vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()


        # Tokenizar la entrada
        input_tokens = tokenizer.encode(user_input)
        input_tensor = torch.tensor([input_tokens])

        # Generar respuesta
        with torch.no_grad():
            output = model(input_tensor)
            output_tokens = torch.argmax(output, dim=-1)[0].tolist()
        
        # Decodificar respuesta
        response = tokenizer.decode(output_tokens)
        return response
    
    def run(self):
        self.chat_interface.display_chat_history()
        user_input = self.chat_interface.get_user_input()

        if user_input:
            self.chat_interface.add_message("user", user_input)
            with st.spinner('Analizando tu consulta...'):
                response = self.get_model_response(user_input)
                self.chat_interface.add_message("assistant", response)