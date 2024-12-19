import torch
from torch.utils.data import Dataset

class LegalDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Codificar todos los textos
        encoded_texts = self.tokenizer.batch_encode(
            texts, 
            max_length=max_length,
            padding=True
        )
        
        for tokens in encoded_texts:
            # No necesitamos añadir START/END tokens manualmente
            # ya que el tokenizer lo hace por nosotros
            self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Input: todos los tokens menos el último
        # Target: todos los tokens menos el primero
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])