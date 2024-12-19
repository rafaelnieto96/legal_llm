import streamlit as st
from frontend.components.chat_interface import ChatInterface
from models.model import LegalLLM
from tokenizer.tokenizer import LegalTokenizer
import torch
from tokenizers import Tokenizer

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
        try:
            # Cargar checkpoint con weights_only=True
            checkpoint = torch.load('best_legal_llm_model.pth', weights_only=True)
            
            # Inicializar tokenizer y cargar su configuración
            tokenizer = LegalTokenizer()
            tokenizer.tokenizer = Tokenizer.from_file('tokenizer/tokenizer.json')
            tokenizer.vocab_size = tokenizer.tokenizer.get_vocab_size()
            
            # Inicializar el modelo con la arquitectura correcta
            model = LegalLLM(
                vocab_size=tokenizer.vocab_size,
                d_model=512,
                nhead=8,
                num_layers=6,
                dropout=0.1
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Cargar los pesos del modelo ignorando las capas extra
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()

            # Procesar input
            input_tokens = tokenizer.encode(user_input)
            input_tensor = torch.tensor([input_tokens]).to('cuda' if torch.cuda.is_available() else 'cpu')

            # Generar respuesta
            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_length=100,
                    temperature=0.7
                )
            
            # Decodificar respuesta
            response = tokenizer.decode(output[0].tolist())
            return response
            
        except Exception as e:
            print(f"Error al generar respuesta: {str(e)}")
            return f"Lo siento, hubo un error al procesar tu consulta: {str(e)}"
    
    def run(self):
        self.chat_interface.display_chat_history()
        user_input = self.chat_interface.get_user_input()

        if user_input:
            self.chat_interface.add_message("user", user_input)
            with st.spinner('Analizando tu consulta...'):
                response = self.get_model_response(user_input)
                self.chat_interface.add_message("assistant", response)