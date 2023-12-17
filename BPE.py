import json

class BPE_Tokenizer:
    def __init__(self):
        self.vocab = set()
        self.token_to_index = {}
        self.index_to_token = {}

    @staticmethod
    def get_stats(vocab):
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                if pair in pairs:
                    pairs[pair] += freq
                else:
                    pairs[pair] = freq
        return pairs

    @staticmethod
    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in v_in:
            w_out = word.replace(bigram, replacement)
            v_out[w_out] = v_in[word]
        return v_out

    @staticmethod
    def get_vocab(text):
        vocab = {}
        for word in text.split():
            word = ' '.join(list(word)) +  ' </w>'
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        return vocab

    def train_until(self, text, vocab_size):
        vocab = self.get_vocab(text)
        self.vocab = set(word for word in vocab for word in word.split())
        # Check if the initial vocabulary size is less than the desired size
        while len(self.vocab) < vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.vocab = set(word for word in vocab for word in word.split())

        # Add special tokens
        self.vocab.add('</u>')

        self.build_index()

    def train(self, text, num_merges):
            vocab = self.get_vocab(text)
            for i in range(num_merges):
                pairs = self.get_stats(vocab)
                if not pairs:
                    break
                best = max(pairs, key=pairs.get)
                vocab = self.merge_vocab(best, vocab)

            self.vocab = set(word for word in vocab for word in word.split())

            # Add special tokens
            self.vocab.add('</u>')

            self.build_index()

    def build_index(self):
        # existing code
        self.token_to_index = {token: index for index, token in enumerate(self.vocab)}
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}

    def tokenize(self, text):
        tokens = []
        for word in text.split():
            subwords = self.get_subwords(word + '</w>')
            tokens.extend(self.token_to_index.get(sw, self.token_to_index['</u>']) for sw in subwords)
        return tokens

    def get_subwords(self, word):
        subwords = []
        while word:
            subword = self.find_longest_subword(word)
            if subword is None:
                subwords.append('</u>')
                break
            subwords.append(subword)
            word = word[len(subword):]
        return subwords

    def find_longest_subword(self, word):
        for i in range(len(word), 0, -1):
            if word[:i] in self.vocab:
                return word[:i]
        return None

    def detokenize(self, token_ids):
        words = []
        current_word = ''
        for token_id in token_ids:
            token = self.index_to_token.get(token_id, '</u>')
            if token == '</w>':
                words.append(current_word)
                current_word = ''
            else:
                current_word += token
        words.append(current_word)
        return ' '.join(words).replace('</w>', ' ')

    def save_vocab(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.token_to_index, f)

    def load_vocab(self, file_path):
        with open(file_path, 'r') as f:
            self.token_to_index = json.load(f)
            self.index_to_token = {int(index): token for token, index in self.token_to_index.items()}
            self.vocab = set(self.token_to_index.keys())
    
    def get_vocab_size(self):
        return len(self.token_to_index)


if __name__ == '__main__':
    # Example usage
    import tqdm as tqdm
    import pandas as pd
    tokenizer = BPE_Tokenizer()
    file_1 = 'data/Harry_Potter_all_books_preprocessed.txt'
    file_2 = 'data/nicolasSTASTrain.txt'
    file_3 = 'data/openwebTrain.txt'
    with open(file_3, 'r') as f:
        full_text = f.read()

        train_text = full_text[:int(0.99*len(full_text))]
    with open(file_2, 'r') as f:
        full_text = f.read()
        test_text = full_text[int(0.99*len(full_text)):]
        # test_text = full_text[int(0.90*len(full_text)):]
    # text = "this is a test sentence for byte pair encoding"

    num_merges =  90
    untils = []
    total_tokens = []
    avg_token_len = []
    avg_token_std = []

    text = train_text+" "+test_text*10000  # we multiplity the test text by to bias the tokenizer towards the test text

    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = text.replace('!', ' ! ')
    text = text.replace('?', ' ? ')
    text = text.replace(':', ' : ')
    text = text.replace(';', ' ; ')
    text = text.replace(')', '')
    text = text.replace('(', '')
    text = text.replace('@', '')
    text = text.replace('|', '')
    text = text.replace(']', '')
    text = text.replace('[', '')
    text = text.replace('~', '')
    text = text.replace('^', '')
    text = text.replace('<', '')
    text = text.replace('>', '')
    text = text.replace('&', '')
    text = text.replace('{', '')
    text = text.replace('}', '')
    text = text.replace('+', '')
    # text = text.replace('-', '')
    text = text.replace('tititi', '')
    text = text.replace('orerer', '')
    text = text.replace('errero', '')
    text = text.replace('\u007f', '')
    text = text.replace('_', '')
    text = text.replace('%', '')
    text = text.replace('$', '')
    text = text.replace('\\', '')
    text = text.replace('=', '')
    text = text.replace('#', '')
    text = text.replace(';', '')
    text = text.replace(':', '')

    #  remove non ascii characters
    text = text.encode("ascii", errors="ignore").decode()


    print("text length:", len(text))
    print("Training...")
    tokenizer.train_until(text, 360)
    # tokenizer.load_vocab('token_to_index.json')
    
    print("Vocab:", tokenizer.token_to_index)
    print("vocab size:", tokenizer.get_vocab_size())

    # Tokenizing
    token_ids = tokenizer.tokenize(test_text)
    print("Token IDs:", token_ids)
    print("num Tokens:", len(token_ids))
    
    # Detokenizing
    detokenized_text = tokenizer.detokenize(token_ids)
    print("Detokenized Text:", detokenized_text)

    # Saving token_to_index mapping
    tokenizer.save_vocab('openweb+Nicolas.json')

