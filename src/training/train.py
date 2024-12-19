import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup

from src.models.model import LegalLLM
from src.tokenizer.tokenizer import LegalTokenizer
from src.data.dataset import LegalDataset

from datasets import load_dataset

def load_training_data(limit=None):
   dataset = load_dataset("Ramitha/spanish-legal-data")
   # texts = list(dataset['train']['Data'])
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

   # Crear directorios para guardar los archivos de entrenamiento
   training_dir = 'training'
   tokenizer_dir = os.path.join(training_dir, 'tokenizer')
   checkpoints_dir = os.path.join(training_dir, 'checkpoints')
   os.makedirs(training_dir, exist_ok=True)
   os.makedirs(tokenizer_dir, exist_ok=True)
   os.makedirs(checkpoints_dir, exist_ok=True)

   # Cargar datos (usar más datos)
   print("Cargando datos...")
   texts = load_training_data(limit=1000)  # Aumentar el límite o quitar para usar todo el dataset

   # Preparar tokenizer
   print("Preparando tokenizer...")
   tokenizer = LegalTokenizer()
   tokenizer.fit(texts)

   # Preparar dataset
   print("Preparando dataset...")
   dataset = LegalDataset(texts, tokenizer, max_length=max_length)
   dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

   # Para guardar el mejor modelo
   best_loss = float('inf')
   
   # Entrenamiento
   print("Iniciando entrenamiento...")
   for epoch in range(epochs):
       model.train()
       total_loss = 0
       progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
       
       for batch_idx, (data, target) in enumerate(progress_bar):
           # Debug info solo para el primer batch del primer epoch
           if epoch == 0 and batch_idx == 0:
               print("\nForma del batch de entrada:", data.shape)
               print("Forma del batch objetivo:", target.shape)
               print("\nPrimeros tokens del primer ejemplo en el batch:")
               decoded_text = tokenizer.decode(data[0][:20].tolist())
               print(decoded_text)
           
           data, target = data.to(device), target.to(device)
           
           optimizer.zero_grad()
           output = model(data)
           loss = criterion(output.view(-1, tokenizer.vocab_size), target.view(-1))
           loss.backward()
           
           # Gradient clipping
           torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
           
           optimizer.step()
           scheduler.step()
           
           total_loss += loss.item()
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