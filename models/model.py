import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

class LegalLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Cargar configuración base del modelo
        self.model_config = AutoConfig.from_pretrained(config.model.model_name)
        
        # Inicializar modelo base
        self.transformer = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            config=self.model_config
        )
        
        # Ajustar el tamaño del vocabulario si es necesario
        self.resize_token_embeddings = self.transformer.resize_token_embeddings
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def save_pretrained(self, path):
        self.transformer.save_pretrained(path)