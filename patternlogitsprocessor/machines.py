from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Self, List, Set, Optional


@dataclass
class TokensInfo:
    N: int  # Number of Tokens
    tids: List[int]  # Token ID List
    ttxs: List[str]  # Token Text List
    tdts: List[str]  # Decoded Token Text List
    eos_id: int


class MachineState(Enum):
    Handling = "Handling"
    Rejected = "Rejected"
    Finished = "Finished"


class MachineLogic(ABC):
    def __init__(self, info: TokensInfo) -> None:
        super().__init__()
        self.info = info

    @abstractmethod
    def put(self,n_token:int, new_tid: Optional[int] = None) -> List[MachineState]:
        pass


class OptionallyRepeatingTokenSet(MachineLogic):
    def __init__(
        self, info: TokensInfo, member_tids: List[int], final_tids: List[int]
    ) -> None:
        super().__init__(info)
        self.member_tids = member_tids
        self.final_tids = final_tids

        self.next_states = [MachineState.Rejected for _ in range(info.N)]
        for tid in member_tids:
            self.next_states[tid] = MachineState.Handling
        for tid in final_tids:
            self.next_states[tid] = MachineState.Finished
    
    def put(self, n_token:int ,new_tid: Optional[int] = None) -> List[MachineState]:
        return self.next_states


class ConstantTokenSequence(MachineLogic):
    def __init__(self, info: TokensInfo, tids: List[int]) -> None:
        super().__init__(info)
        self.next_states_s = []
        for i,tid in enumerate(tids[:-1]):
            next_state = [MachineState.Rejected for _ in range(info.N)]
            next_state[tid] = MachineState.Handling
            self.next_states_s.append(next_state)
        next_state = [MachineState.Rejected for _ in range(info.N)]
        next_state[tids[-1]] = MachineState.Finished
        self.next_states_s.append(next_state)

    def put(self,n_token:int, new_tid: Optional[None] = None) -> List[MachineState]:
        return self.next_states_s[n_token]
