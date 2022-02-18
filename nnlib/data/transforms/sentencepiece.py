from typing import Dict, Any, List
import sentencepiece as stp
import torch

from nnlib.data.transforms import register_transform
from nnlib.data.transforms.transform import BaseTransform
from nnlib.utils.print_utils import print_log


@register_transform("SentencePieceWrapper")
class SentencePieceWrapper(BaseTransform):
    """SentencePiece, mostly used for language modeling and speech recognition.
    Should be already processed and vocabulary file *.model exist.
    """

    def __init__(self,
                 vocab_model: str, *,
                 add_bos: bool = True,
                 add_eos: bool = True,
                 enable_sampling: bool = False,
                 alpha: float = 0.1):
        super(SentencePieceWrapper, self).__init__()
        if vocab_model[-6:] != ".model":
            raise ValueError(f"[ERROR:DATA] SentencePiece model {vocab_model} is invalid.")

        self.tokenizer = stp.SentencePieceProcessor()
        self.tokenizer.Init(model_file=vocab_model, add_bos=add_bos, add_eos=add_eos,
                            enable_sampling=enable_sampling, alpha=alpha)

        self.unk_piece = self.tokenizer.IdToPiece(self.tokenizer.unk_id())
        print_log(f"[LOG:DATA] SentencePiece unk: {self.unk_piece} (id: {self.tokenizer.unk_id()})")

    def encode(self, utterance: str, sampling=None) -> List[int]:
        utterance = utterance.replace(self.unk_piece, "⁇")  # pre-defined one
        encoding = self.tokenizer.Encode(utterance, enable_sampling=sampling)
        return encoding

    def decode(self, sequence: List[int]) -> str:
        decoding: str = self.tokenizer.Decode(sequence)
        decoding = decoding.replace("??", self.unk_piece).replace("⁇", self.unk_piece).strip()
        return decoding

    def __call__(self, utterance: str, sampling=None) -> torch.Tensor:
        encoding = self.encode(utterance, sampling=sampling)
        return torch.tensor(encoding, dtype=torch.long)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(
            vocab_model=config.get("vocab_model"),
            add_bos=config.get("add_bos", True),
            add_eos=config.get("add_eos", True),
            enable_sampling=config.get("enable_sampling", False),
            alpha=config.get("alpha", 0.1)
        )
