import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pathlib import Path

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LegalLLM:
    def __init__(self, model_name='gpt2', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Utilizando dispositivo: {self.device}")
        
        # Inicializar tokenizer y modelo
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise
        
    def generate_response(self, prompt, max_length=100):
        try:
            # Tokenizar input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generar respuesta
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decodificar respuesta
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logging.error(f"Error en la generación: {str(e)}")
            return None

def main():
    try:
        # Inicializar el modelo
        llm = LegalLLM()
        
        # Ejemplo de uso
        prompt = "¿Cuáles son los requisitos para presentar una demanda civil?"
        response = llm.generate_response(prompt)
        print(f"Respuesta: {response}")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")

if __name__ == "__main__":
    main()