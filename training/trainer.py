import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path

class LegalModelTrainer:
    def __init__(self, model, config, train_dataset, val_dataset=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.setup_training()
        
    def setup_training(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True
        )
        
        if self.val_dataset:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.model.batch_size
            )
            
        # Optimizador
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.model.warmup_steps,
            num_training_steps=len(self.train_dataloader) * self.config.model.epochs
        )
        
    def train(self):
        self.model.to(self.device)
        
        for epoch in range(self.config.model.epochs):
            self.model.train()
            total_loss = 0
            
            with tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}') as pbar:
                for step, batch in enumerate(pbar):
                    # Mover batch a GPU/CPU
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    loss = outputs.loss / self.config.model.gradient_accumulation_steps
                    total_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Actualizar pesos
                    if (step + 1) % self.config.model.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                    # Actualizar progress bar
                    pbar.set_postfix({'loss': total_loss / (step + 1)})
                    
                    # Evaluación periódica
                    if step % self.config.training.eval_steps == 0:
                        self.evaluate()
                        
            # Guardar checkpoint
            self.save_checkpoint(epoch)
            
    def evaluate(self):
        if not self.val_dataset:
            return
            
        self.model.eval()
        total_eval_loss = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                total_eval_loss += outputs.loss.item()
                
        avg_eval_loss = total_eval_loss / len(self.val_dataloader)
        logging.info(f"Validation Loss: {avg_eval_loss}")
        
    def save_checkpoint(self, epoch):
        checkpoint_dir = self.config.training.output_dir / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        logging.info(f"Modelo guardado en {checkpoint_dir}")