from typing import Dict, Any, Tuple, List, Optional
import torch

from nnlib.data.transforms import register_transform
from nnlib.data.transforms.transform import BaseTransform
from nnlib.data.helpers.dictionary import ENGLISH_GRAPHEMES, Dictionary


@register_transform("GraphemeWrapper")
class GraphemeWrapper(BaseTransform):

    def __init__(self,
                 vocab: Optional[Tuple[str]] = None, *,
                 add_bos: bool = True,
                 add_eos: bool = True,
                 pad_token: str = "<b>",
                 bos_token: str = "<s>",
                 eos_token: str = "</s>",
                 lowercase: bool = False):
        super(GraphemeWrapper, self).__init__()
        if vocab is None:
            vocab = ENGLISH_GRAPHEMES

        if lowercase:
            vocab = tuple(v.lower() for v in vocab)
        else:
            vocab = tuple(v.upper() for v in vocab)

        self.lowercase = lowercase
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # handle special tokens
        # order: pad, bos, eos, ...
        if add_eos:
            vocab = (eos_token,) + vocab
        if add_bos:
            vocab = (bos_token,) + vocab
        vocab = (pad_token,) + vocab

        self.vocab = vocab
        self.dictionary = Dictionary(bos=bos_token, pad=pad_token, eos=eos_token)
        for c in vocab:
            self.dictionary.add_token(c, n=0)
        self.dictionary.finalize()  # order will be preserved.

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, utterance: str) -> List[int]:
        utterance = utterance.strip().replace(" ", "_")
        if self.lowercase:
            utterance = utterance.lower()
        else:
            utterance = utterance.upper()

        chars = [c for c in utterance]
        if self.add_bos:
            chars = [self.bos_token] + chars
        if self.add_eos:
            chars = chars + [self.eos_token]

        indices = [self.dictionary.get_token_idx(c, use_unknown=False) for c in chars]
        return indices

    def decode(self, sequence: List[int]) -> str:
        chars = [self.dictionary.get_idx_token(i) for i in sequence]
        utterance = "".join(chars).replace("_", " ")
        utterance = utterance.replace(self.pad_token, "").replace(self.bos_token, "").replace(self.eos_token, "")
        return utterance

    def __call__(self, utterance: str) -> torch.Tensor:
        indices = self.encode(utterance)
        return torch.tensor(indices, dtype=torch.long)

    @classmethod
    def from_config(cls, config: Dict[str, Any], vocab=None):
        return cls(
            vocab=config.get("vocab", vocab),
            add_bos=config.get("add_bos", True),
            add_eos=config.get("add_eos", True),
            pad_token=config.get("pad_token", "<b>"),
            bos_token=config.get("bos_token", "<s>"),
            eos_token=config.get("eos_token", "</s>"),
            lowercase=config.get("lowercase", False),
        )
