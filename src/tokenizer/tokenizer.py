from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import torch

class LegalTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Configurar tokens especiales
        self.special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        
        # Mapeo de tokens especiales a IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.start_token_id = 2
        self.end_token_id = 3

    def fit(self, texts):
        # Configurar el trainer con los tokens especiales
        trainer = BpeTrainer(
            special_tokens=self.special_tokens,
            vocab_size=32000,
            min_frequency=2
        )
        
        # Entrenar el tokenizer
        self.tokenizer.train_from_iterator(texts, trainer)
        
        # Configurar post-procesamiento para añadir tokens de inicio/fin
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.start_token} $A {self.end_token}",
            special_tokens=[
                (self.start_token, self.start_token_id),
                (self.end_token, self.end_token_id),
            ],
        )
        
        # Actualizar vocab_size
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        """Convierte texto a IDs de tokens"""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids):
        """Convierte IDs de tokens a texto"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Filtrar tokens especiales si es necesario
        filtered_ids = [id for id in ids if id not in [self.pad_token_id]]
        
        return self.tokenizer.decode(filtered_ids)

    def batch_encode(self, texts, max_length=None, padding=True):
        """Codifica un batch de textos"""
        encodings = self.tokenizer.encode_batch(texts)
        
        if max_length and padding:
            # Padding/truncating a una longitud fija
            return self._pad_or_truncate([e.ids for e in encodings], max_length)
        
        return [e.ids for e in encodings]

    def _pad_or_truncate(self, sequences, max_length):
        """Aplica padding o truncamiento a una lista de secuencias"""
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) > max_length:
                padded_sequences.append(seq[:max_length])
            else:
                padded_sequences.append(seq + [self.pad_token_id] * (max_length - len(seq)))
                
        return padded_sequences
    
    def save(self, path):
        """Guarda el tokenizer en un archivo"""
        self.tokenizer.save(path)
    
    def load(self, path):
        """Carga el tokenizer desde un archivo"""
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()