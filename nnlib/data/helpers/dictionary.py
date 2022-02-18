from typing import Optional, List
from collections import OrderedDict, Counter


class Dictionary(object):

    def __init__(self, *,
                 bos="<s>",
                 pad="<pad>",
                 eos="</s>",
                 unk="<unk>",
                 mask="<mask>",
                 extra_special_tokens: Optional[dict] = None):
        self.special_tokens = {"bos": bos, "pad": pad, "eos": eos, "unk": unk, "mask": mask}
        if extra_special_tokens is not None:
            self.special_tokens.update(extra_special_tokens)

        self.token_to_idx = OrderedDict()  # {"the": 0, "some": 1, ... }
        self.counter = Counter()  # {"the": 1238, "some": 978, ...}
        self.idx_to_token = list()
        self._is_finalized = False

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def add_special_token(self, token_type: str = "unk") -> int:
        if self._is_finalized:
            raise ValueError("[ERROR:DICT] Dictionary is finalized, can't add special token.")
        if token_type not in self.special_tokens:
            raise ValueError(f"[ERROR:DICT] Token type is {token_type} not included in special tokens.")
        idx = self.add_token(self.special_tokens[token_type], n=0)
        return idx

    def get_special_token(self, token_type: str = "unk") -> str:
        if token_type not in self.special_tokens:
            raise ValueError(f"[ERROR:DICT] Token type {token_type} is not included in special tokens.")
        return self.special_tokens[token_type]

    def add_token(self, token, n: int = 1) -> int:
        if self._is_finalized:
            raise ValueError("[ERROR:DICT] Dictionary is finalized, can't add token.")

        if token in self.token_to_idx:  # already exist
            idx = self.token_to_idx[token]
            self.counter[token] += n
        else:  # create new token entry
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.counter[token] = n
        return idx

    def get_token_idx(self, token: str, use_unknown: bool = True) -> int:
        unknown_token = self.special_tokens["unk"]
        if use_unknown and (unknown_token not in self.token_to_idx):
            raise ValueError("[ERROR:DICT] You should first register <unk> to dictionary by add_special_token('unk').")

        if token not in self.token_to_idx:
            if use_unknown:
                token = unknown_token  # <unk>
            else:
                raise KeyError(f"[ERROR:DICT] Token {token} is not in Dictionary.")
        return self.token_to_idx[token]

    def get_idx_token(self, idx: int) -> str:
        try:
            return self.idx_to_token[idx]
        except IndexError:
            raise IndexError(f"[ERROR:DICT] Index {idx} invalid for length {len(self.idx_to_token)}.")

    def finalize(self,
                 min_count_threshold: int = 0,
                 max_num_words_threshold: int = -1,
                 pad_to_multiple: int = 1,
                 force_special_token_index: Optional[List] = None):
        """
        If some special tokens should be in certain order, set special_token_index to be first appear.
        ex) ["pad", "unk", "eos"]  the index of each will be 0, 1, 2, respectively.
        """
        # sort by frequency, apply thresholding
        if max_num_words_threshold < 0:
            max_num_words_threshold = len(self)

        new_token_to_idx = OrderedDict()
        new_counter = Counter()
        new_idx_to_token = list()

        if force_special_token_index is None:
            force_special_token_index = []

        for t in force_special_token_index:
            if t not in self.counter:
                raise ValueError(f"[ERROR:DICT] Special token {t} is not added before.")
            new_token_to_idx[t] = len(new_token_to_idx)
            new_counter[t] = self.counter[t] if (t in self.counter) else 0
            new_idx_to_token.append(t)

        for token, count in self.counter.most_common(n=max_num_words_threshold + len(force_special_token_index)):
            if (force_special_token_index is not None) and (token in force_special_token_index):
                continue  # already added

            if len(new_token_to_idx) >= max_num_words_threshold:
                break

            if count >= min_count_threshold:
                new_token_to_idx[token] = len(new_token_to_idx)
                new_counter[token] = count
                new_idx_to_token.append(token)

        self.token_to_idx = new_token_to_idx
        self.counter = new_counter
        self.idx_to_token = new_idx_to_token
        assert len(self.token_to_idx) == len(self.counter) == len(self.idx_to_token)

        if pad_to_multiple > 1:
            i = 0
            while len(self) % pad_to_multiple != 0:
                dummy_token = f"dummy{i:03d}"
                assert dummy_token not in self.token_to_idx
                self.add_token(dummy_token, n=0)
                i += 1

        self._is_finalized = True

    def state_dict(self) -> dict:
        if not self._is_finalized:
            raise ValueError("[ERROR:DICT] Trying to get state_dict from not-finalized dictionary.")

        state_dict = dict()
        state_dict["token_to_idx"] = self.token_to_idx
        state_dict["counter"] = self.counter
        state_dict["idx_to_token"] = self.idx_to_token
        state_dict["special_tokens"] = self.special_tokens
        return state_dict

    def load_state_dict(self, state_dict: dict):
        if self._is_finalized:
            raise ValueError("[ERROR:DICT] Trying to load state_dict to finalized dictionary.")

        self.token_to_idx = state_dict["token_to_idx"]
        self.counter = state_dict["counter"]
        self.idx_to_token = state_dict["idx_to_token"]
        self.special_tokens = state_dict["special_tokens"]
        self._is_finalized = True

    def __repr__(self) -> str:
        s = f"Dictionary (vocabulary size: {len(self)})\n"
        for token, idx in self.token_to_idx.items():
            count = self.counter[token]
            s += f"{idx:<8}\t{token:<32}\t{count:<8}\n"
        return s


ENGLISH_CHARACTERS = (
    "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
)

ENGLISH_GRAPHEMES = ENGLISH_CHARACTERS + ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "'")

ENGLISH_GRAPHEMES_V2 = ENGLISH_CHARACTERS + ("'",)
