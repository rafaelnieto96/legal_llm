import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from src.models.model import LegalLLM
from src.tokenizer.tokenizer import LegalTokenizer
from src.data.dataset import LegalDataset

from datasets import load_dataset

def load_training_data(limit=None):
    dataset = load_dataset("Ramitha/spanish-legal-data")
    texts = list(dataset['train']['Data'][:10]) 
    if limit:
        texts = texts[:limit]
    
    print("Ejemplo de los primeros 2 textos:")
    print(texts[0][:200])
    print("\n---\n")
    print(texts[1][:200])
    
    return texts

def train():
    # Parámetros mejorados
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    epochs = 50
    learning_rate = 5e-5
    max_length = 512
    warmup_steps = 4000
    gradient_clip_val = 1.0
    accumulation_steps = 4  # Nuevo: pasos de acumulación

    # Crear directorios para guardar los archivos de entrenamiento
    training_dir = 'train_files'
    tokenizer_dir = os.path.join(training_dir, 'tokenizer')
    checkpoints_dir = os.path.join(training_dir, 'checkpoints')
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Cargar datos
    print("Cargando datos...")
    texts = load_training_data(limit=100)

    # Preparar tokenizer
    print("Preparando tokenizer...")
    tokenizer = LegalTokenizer()
    tokenizer.fit(texts)

    # Preparar dataset
    print("Preparando dataset...")
    dataset = LegalDataset(texts, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # Paralelización de carga
        pin_memory=True,  # Optimización para GPU
        persistent_workers=True  # Mantiene workers vivos entre épocas
    )

    # Crear modelo
    print("Creando modelo...")
    model = LegalLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.1
    ).to(device)

    # Optimizador y scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * len(dataloader)
    )

    # Inicializar scaler para mixed precision
    scaler = GradScaler()

    # Para guardar el mejor modelo
    best_loss = float('inf')
    
    # Entrenamiento
    print("Iniciando entrenamiento...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        optimizer.zero_grad()  # Movido fuera del bucle interno
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            # Mixed precision training
            with autocast():
                output = model(data)
                loss = criterion(output.view(-1, tokenizer.vocab_size), target.view(-1))
                loss = loss / accumulation_steps  # Normalizar pérdida
            
            # Backward pass con gradient accumulation
            scaler.scale(loss).backward()
            
            # Actualizar solo cada accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item() * accumulation_steps  # Ajustar loss para logging
            
            progress_bar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'lr': scheduler.get_last_lr()[0]
            })

        avg_loss = total_loss/len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        # Guardar el mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Guardando mejor modelo con loss: {best_loss}")
            
            # Guardar el tokenizer
            tokenizer.tokenizer.save(os.path.join(tokenizer_dir, 'tokenizer.json'))
            
            # Guardar el modelo
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'vocab_size': tokenizer.vocab_size,
            }, os.path.join(training_dir, 'best_legal_llm_model.pth'))

        # Guardar checkpoint cada 5 epochs
        if (epoch + 1) % 5 == 0:
            # Guardar el tokenizer
            tokenizer.tokenizer.save(
                os.path.join(tokenizer_dir, f'tokenizer_checkpoint_{epoch+1}.json')
            )
            
            # Guardar el modelo
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': tokenizer.vocab_size,
            }, os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    train()