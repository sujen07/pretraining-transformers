import json
import re
from collections import Counter, defaultdict
import time
import gc
import os
# GPT-2-style pretokenization: optional leading space on words/numbers/punctuation.
_PRETOKENIZE_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",
    re.UNICODE,
)

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
    def __init__(self, target_vocab_size: int, chunk_size: int, dataset_path: str, tokenizer_path: str):
        self.target_vocab_size = target_vocab_size
        self.dataset_path = dataset_path
        self.tokenizer_path = tokenizer_path
        self.corpus = []
        self.chunk_size = chunk_size
        self.vocab = set(SPECIAL_TOKENS)
        self.vocab_trie = Trie()

        if os.path.exists(self.tokenizer_path):
            self._load_vocab(self.tokenizer_path)
        else:
            self._train()


    def _initialize_corpus(self):
        with open(self.dataset_path, "r") as f:
            for line in f:
                text = json.loads(line)["text"]
                tokens = [bytes(token, "utf-8") for token in text]
                self.corpus.append(tokens)
                for token in tokens:
                    self.vocab.add(token)
    
    def _count_pairs(self, chunk_idx: int = 0, chunk_size: int = 10000):
        self.pair_positions = None
        gc.collect()
   
        self.pair_positions = defaultdict(set)
        for i, sequence in enumerate(self.corpus):
            if i < chunk_idx:
                continue
            if i >= chunk_idx + chunk_size:
                break
            for j in range(len(sequence) - 1):
                if sequence[j] is None or sequence[j + 1] is None:
                    continue
                self.pair_positions[(sequence[j], sequence[j + 1])].add((i, j))

    def _remove_pair_position(self, pair: tuple, seq_idx: int, pos_idx: int) -> None:
        positions = self.pair_positions.get(pair)
        if not positions:
            return
        positions.discard((seq_idx, pos_idx))
        if not positions:
            del self.pair_positions[pair]

    def _train(self):
        """
        Train the BPE model.
        """

        start_time = time.time()
        self._initialize_corpus()
        end_time = time.time()
        print(f"Time taken for initializing corpus: {end_time - start_time} seconds")

        start_time = time.time()
        chunk_num = len(self.corpus) // self.chunk_size

        for chunk_idx in range(chunk_num):
            start_time = time.time()
            self._count_pairs(chunk_idx, self.chunk_size)
            end_time = time.time()
            print(f"Time taken for counting pairs: {end_time - start_time} seconds")
            merges_per_chunk = self.target_vocab_size // chunk_num
            for _ in range(merges_per_chunk):
                if not self.pair_positions:
                    break
                print(f"Training BPE: {len(self.vocab)} / {self.target_vocab_size}")
                pair = max(self.pair_positions, key=lambda p: len(self.pair_positions[p]))
                all_pair_indices = self.pair_positions.pop(pair)
                new_token = pair[0] + pair[1]
                self.vocab.add(new_token)
                for seq_idx, token_idx in all_pair_indices:
                    sequence = self.corpus[seq_idx]
                    if sequence[token_idx] != pair[0] or sequence[token_idx + 1] != pair[1]:
                        continue

                    left_idx = token_idx - 1 if token_idx > 0 else None
                    while left_idx is not None and left_idx >= 0 and sequence[left_idx] is None:
                        left_idx -= 1
                    if left_idx is not None and left_idx < 0:
                        left_idx = None
                    left = sequence[left_idx] if left_idx is not None else None

                    right_idx = token_idx + 2 if token_idx + 2 < len(sequence) else None
                    while right_idx is not None and right_idx < len(sequence) and sequence[right_idx] is None:
                        right_idx += 1
                    if right_idx is not None and right_idx >= len(sequence):
                        right_idx = None
                    right = sequence[right_idx] if right_idx is not None else None

                    if left is not None:
                        self._remove_pair_position((left, pair[0]), seq_idx, left_idx)
                    if right is not None:
                        self._remove_pair_position((pair[1], right), seq_idx, token_idx + 1)

                    sequence[token_idx] = new_token
                    sequence[token_idx + 1] = None

                    if left is not None:
                        self.pair_positions[(left, new_token)].add((seq_idx, left_idx))
                    if right is not None:
                        self.pair_positions[(new_token, right)].add((seq_idx, token_idx))

        end_time = time.time()
        print(f"Time taken for training: {end_time - start_time} seconds")
        
        self._save_vocab(self.tokenizer_path)
        print(f"Vocab saved to {self.tokenizer_path}")
        self._load_vocab(self.tokenizer_path)

        # Garbage collection
        self.corpus = None
        self.pair_positions = None
        gc.collect()

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
    bpe = BPE(target_vocab_size=10000, chunk_size=10000, dataset_path="src/data/openwebtext-100k.jsonl", tokenizer_path="src/tokenization/bpe_tokenizer.json")
    start_time = time.time()
    bpe._train()
    end_time = time.time()
    print(f"Time taken for training: {end_time - start_time} seconds")
   
    # print(bpe.encode("Hello, world!"))
    # print(bpe.decode([1, 2, 3, 4]))
    # print(bpe.encode("Hello, world!"))
    # print(bpe.decode([1, 2, 3, 4]))
    # print(bpe.encode("Hello, world!"))