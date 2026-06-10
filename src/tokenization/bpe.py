import json
from collections import defaultdict

# Special tokens stored as bytes in the vocab
BOS_TOKEN = b"<bos>"
EOS_TOKEN = b"<eos>"
UNK_TOKEN = b"<unk>"
PAD_TOKEN = b"<pad>"
SPECIAL_TOKENS = {BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN}

class TrieNode:
    def __init__(self):
        # A dictionary map handles children elegantly and saves space over a fixed array.
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: bytes) -> None:
        """Inserts a word into the trie."""
        current = self.root
        for byte in word:
            if byte not in current.children:
                current.children[byte] = TrieNode()
            current = current.children[byte]
        current.is_end_of_word = True

    def search(self, word: bytes) -> bool:
        """Returns True if the word is in the trie."""
        current = self.root
        for byte in word:
            if byte not in current.children:
                return False
            current = current.children[byte]
        return current.is_end_of_word

    def startsWith(self, prefix: bytes) -> bool:
        """Returns True if there is any word in the trie that starts with the given prefix."""
        current = self.root
        for byte in prefix:
            if byte not in current.children:
                return False
            current = current.children[byte]
        return True
    

class BPE:
    def __init__(self, target_vocab_size: int, dataset_path: str, tokenizer_path: str):
        self.corpus = []
        self.target_vocab_size = target_vocab_size
        self.dataset_path = dataset_path
        self.vocab = set(SPECIAL_TOKENS)
        self.vocab_trie = Trie()
        self._initialize_corpus()
        self._train()
        self._save_vocab(tokenizer_path)

    def _initialize_corpus(self):
        """
        Initialize the vocabulary.
        """
        with open(self.dataset_path, 'r') as f:
            for line in f:
                text = json.loads(line)["text"]
                word = [bytes([b]) for b in text.encode("utf-8")]
                self.corpus.append(word)
                self.vocab.update(word)

    def _count_pairs(self):
        """
        Count the frequency of pairs of tokens in the corpus.
        """
        pair_positions = defaultdict(set)
        for word_idx, word in enumerate(self.corpus):
            for i in range(len(word) - 1):
                pair_positions[(word[i], word[i + 1])].add((word_idx, i))
        return pair_positions

    def _train(self):
        """
        Train the BPE tokenizer.
        """
        pair_positions = self._count_pairs()

        while len(self.vocab) < self.target_vocab_size:
            pair = max(
                (k for k, v in pair_positions.items() if v),
                key=lambda x: len(pair_positions[x]),
            )
            new_token = pair[0] + pair[1]
            self.vocab.add(new_token)

            for word_idx, i in sorted(pair_positions[pair], key=lambda x: (x[0], -x[1])):
                word = self.corpus[word_idx]
                if i >= len(word) - 1 or (word[i], word[i + 1]) != pair:
                    continue

                if i > 0:
                    left_pair = (word[i - 1], pair[0])
                    pair_positions[left_pair].discard((word_idx, i - 1))
                    pair_positions[(word[i - 1], new_token)].add((word_idx, i - 1))

                if i + 2 < len(word):
                    right_pair = (pair[1], word[i + 2])
                    pair_positions[right_pair].discard((word_idx, i + 1))
                    pair_positions[(new_token, word[i + 2])].add((word_idx, i))

                word[i] = new_token
                del word[i + 1]

            del pair_positions[pair]

        base_tokens = sorted(self.vocab - SPECIAL_TOKENS)
        sorted_vocab = sorted(SPECIAL_TOKENS) + base_tokens

        for token in sorted_vocab:
            self.vocab_trie.insert(token)

        self.id_to_word = dict(enumerate(sorted_vocab))
        self.word_to_id = {word: i for i, word in enumerate(sorted_vocab)}


    def _save_vocab(self, path: str):
        with open(path, "w") as f:
            json.dump([t.decode("latin-1") for t in sorted(self.vocab)], f)

    def _load_vocab(self, path: str):
        with open(path) as f:
            sorted_vocab = [t.encode("latin-1") for t in json.load(f)]
        self.vocab = set(sorted_vocab)
        self.id_to_word = dict(enumerate(sorted_vocab))
        self.word_to_id = {w: i for i, w in enumerate(sorted_vocab)}
        self.vocab_trie = Trie()
        for token in sorted_vocab:
            self.vocab_trie.insert(token)
            


    def encode(self, text: str):
        """
        Encode the text into tokens.
        """
        data = text.encode("utf-8")
        tokens = []
        l = 0
        while l < len(data):
            node = self.vocab_trie.root
            best_end = None
            for r in range(l, len(data)):
                if data[r] not in node.children:
                    break
                node = node.children[data[r]]
                if node.is_end_of_word:
                    best_end = r + 1
            if best_end is None:
                tokens.append(self.word_to_id[UNK_TOKEN])
                l += 1
            else:
                tokens.append(self.word_to_id[data[l:best_end]])
                l = best_end
        return tokens

        
    
    def decode(self, tokens: list[int]):
        """
        Decode the tokens into text.
        """
        return b"".join(self.id_to_word[token] for token in tokens).decode("utf-8")


if __name__ == "__main__":
    bpe = BPE(target_vocab_size=10000, dataset_path="src/data/openwebtext-100k.jsonl", tokenizer_path="src/tokenization/bpe_tokenizer.json")

    print(bpe.encode("Hello, world!"))
    print(bpe.decode([1, 2, 3, 4]))
    print(bpe.encode("Hello, world!"))
    print(bpe.decode([1, 2, 3, 4]))
    print(bpe.encode("Hello, world!"))