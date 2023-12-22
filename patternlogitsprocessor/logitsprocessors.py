from typing import List
from torch import FloatTensor, LongTensor, tensor, no_grad
from transformers import LogitsProcessor, PreTrainedTokenizer
from patternlogitsprocessor.machines import MachineState, MachineLogic


class PatternLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        logics: List[MachineLogic],
    ) -> None:
        super().__init__()
        self.logics = logics
        self.n_logic = len(logics)
        self.logic_idx = 0
        self.n_token_on_logic_firstcall = 0
        self.next_states = None
    
    @no_grad
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        input_ids_numpy = input_ids.detach().numpy()
        n_token = input_ids_numpy.shape[1]
        new_tid = int(input_ids_numpy[0, -1])

        # Update the current state
        if self.next_states:
            state = self.next_states[new_tid]
        else:
            state = MachineState.Handling

        # Check state
        if state == MachineState.Finished:
            # Go to the next machine
            self.logic_idx += 1
            self.n_token_on_logic_firstcall = n_token

        # Get the next state mapping
        self.n_token_on_logic_firstcall = self.n_token_on_logic_firstcall or n_token
        n_token_on_logic = n_token - self.n_token_on_logic_firstcall
        self.next_states = self.logics[self.logic_idx].put(n_token_on_logic, new_tid)

        # Return scores where all indexes with Rejected state -inf
        scores_next = tensor(
            [[
                (float("-inf") if state == MachineState.Rejected else score)
                for (state,score) in zip(self.next_states,scores[0].numpy())
            ]]
        )
        return scores_next
