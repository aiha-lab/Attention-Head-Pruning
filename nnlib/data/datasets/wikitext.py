from typing import Optional, Any, Dict
import os
import torch

from nnlib.data.datasets import register_dataset
from nnlib.data.datasets.dataset import HappyDataset
from nnlib.data.helpers.dictionary import Dictionary
from nnlib.data.helpers.utils import tokenize_line_by_word
from nnlib.utils.print_utils import print_log
from nnlib.utils.dist_utils import is_master


class HappyWikitext(HappyDataset):
    """Wikitext for word-level language modeling"""
    _WIKITEXT_TYPE = None

    def __init__(self,
                 data_dir: str,
                 mode: str = "train",
                 preprocess_dict_path: Optional[str] = None):
        super(HappyWikitext, self).__init__()
        # ---------------------------------------------------------------- #
        mode = mode.lower()
        if mode not in ("train", "valid", "test"):
            raise ValueError(f"[ERROR:DATA] {self._WIKITEXT_TYPE} dataset mode {mode} is unsupported.")

        self.mode = mode
        data_path = os.path.join(data_dir, f"wiki.{mode}.tokens")

        self.dictionary = Dictionary()
        self.data = []

        # ---------------------------------------------------------------- #
        # dictionary
        if preprocess_dict_path is not None:
            self.dictionary.load_state_dict(torch.load(preprocess_dict_path))
        else:
            cached_dictionary_path = os.path.join(data_dir, "dictionary.pth")
            if os.path.isfile(cached_dictionary_path):
                self.dictionary.load_state_dict(torch.load(cached_dictionary_path))
            else:  # run for the first time
                print(f"[LOG:DATA] Creating dictionary for Wikitext {mode}.")
                if self.mode != "train":
                    raise ValueError("[ERROR:DATA] Dictionary should be made by train helpers.")
                # WikiText use two special tokens
                self.dictionary.add_special_token("unk")  # <unk>
                self.dictionary.add_special_token("eos")  # </s>

                with open(data_path, "r", encoding="utf-8") as f:
                    eos_token = self.dictionary.get_special_token("eos")
                    for line in f:
                        words = tokenize_line_by_word(line) + [eos_token]
                        for word in words:
                            self.dictionary.add_token(word)
                self.dictionary.finalize(force_special_token_index=[
                    self.dictionary.get_special_token("unk"),
                    self.dictionary.get_special_token("eos"),
                ])

                if is_master():
                    torch.save(self.dictionary.state_dict(), cached_dictionary_path)

        # ---------------------------------------------------------------- #
        # data
        cached_data_path = os.path.join(data_dir, f"{mode}.cache.pth")
        if os.path.isfile(cached_data_path):
            self.data = torch.load(cached_data_path)
        else:
            print(f"[LOG:DATA] Creating cache data for Wikitext {mode}.")
            with open(data_path, "r", encoding="utf_8") as f:
                eos_token = self.dictionary.get_special_token("eos")
                for line in f:
                    words = tokenize_line_by_word(line) + [eos_token]
                    for word in words:
                        # unknown is already preprocessed in wikitext.
                        word_idx = self.dictionary.get_token_idx(word, use_unknown=False)
                        self.data.append(word_idx)

            if is_master():
                torch.save(self.data, cached_data_path)

        # ---------------------------------------------------------------- #
        # check
        if self._WIKITEXT_TYPE == "wikitext-2":
            required_vocab_size = 33278
        elif self._WIKITEXT_TYPE == "wikitext-103":
            required_vocab_size = 267735
        else:
            raise ValueError("[ERROR:DATA] Unsupported Wikitext type")
        if len(self.dictionary) != required_vocab_size:
            raise ValueError(f"[ERROR:DATA] Dictionary have {len(self.dictionary)} tokens, "
                             f"but it should have {required_vocab_size} for {self._WIKITEXT_TYPE}.")

        s = f"[LOG:LM] {self._WIKITEXT_TYPE} {self.mode}: {len(self.data)} words, {len(self.dictionary)} vocabulary."
        print_log(s)

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        # should not be used.
        return self.data[index]

    @classmethod
    def from_config(cls, config: Dict[str, Any], transform=None, target_transform=None) -> "HappyWikitext":
        return cls(
            data_dir=config.get("data_dir"),
            mode=config.get("mode", "train"),
            preprocess_dict_path=config.get("preprocess_dict_path", None)
        )


@register_dataset("HappyWikitext2")
class HappyWikitext2(HappyWikitext):
    _WIKITEXT_TYPE = "wikitext-2"


@register_dataset("HappyWikitext103")
class HappyWikitext103(HappyWikitext):
    _WIKITEXT_TYPE = "wikitext-103"
