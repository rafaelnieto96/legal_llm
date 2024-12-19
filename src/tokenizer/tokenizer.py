class SimpleTokenizer:
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.vocab_size = 4

    def fit(self, texts):
        words = set()
        for text in texts:
            words.update(text.split())
        
        for word in sorted(words):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text):
        return [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                for word in text.split()]

    def decode(self, indices):
        return ' '.join([self.idx_to_word[idx] for idx in indices])