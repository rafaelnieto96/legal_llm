import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.models.model import LegalLLM
from src.tokenizer.tokenizer import SimpleTokenizer
from src.data.dataset import LegalDataset

def load_training_data(data_dir):
    texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

def train():
    # Parámetros
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 10
    learning_rate = 0.001

    # Cargar datos
    print("Cargando datos...")
    texts = load_training_data('data/training_texts')

    # Preparar tokenizer
    print("Preparando tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)

    # Preparar dataset
    print("Preparando dataset...")
    dataset = LegalDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Crear modelo
    print("Creando modelo...")
    model = LegalLLM(tokenizer.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento
    print("Iniciando entrenamiento...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, tokenizer.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss/(batch_idx+1)})

        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

    # Guardar modelo
    print("Guardando modelo y tokenizador...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.word_to_idx
    }, 'legal_llm_model.pth')
    
if __name__ == "__main__":
    train()