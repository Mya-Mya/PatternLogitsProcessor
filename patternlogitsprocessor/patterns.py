from patternlogitsprocessor.machines import *
from typing import List, Dict
import re
from transformers import PreTrainedTokenizer
from dataclasses import dataclass

def build_machines(tokenizer:PreTrainedTokenizer,scheme:List[Dict[str,str]])->List[MachineLogic]:
    vocab = tokenizer.get_vocab()
    N = len(vocab)
    info = TokensInfo(
        N,
        list(range(N)),
        list(vocab.keys()),
        tokenizer.batch_decode(range(N))
    )
    assert len(scheme)>0
    if len(scheme)==1:
        return build_machines(
            tokenizer,
            scheme+[ConstantTokenSequence(info,[tokenizer.eos_token_id])]
        )
    # TODO