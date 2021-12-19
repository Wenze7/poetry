from transformers import GPT2LMHeadModel
from torch.nn import CrossEntropyLoss


class MYGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()

    def forward2(self, input_ids,
                 labels=None,
                 token_type_ids=None,
                 content_id=None,
                 layer_past=None,
                 output_hidden_states=None):

        outputs = self.forward(input_ids=input_ids,
                               labels=labels,
                               past_key_values=layer_past,
                               output_hidden_states=output_hidden_states)
        if labels != None:
            if token_type_ids != None:
                mask = (token_type_ids == content_id).long()
                labels = labels * mask

            loss, logits = outputs[:2]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            num = shift_labels.ne(0).long().sum().item()

            loss = loss / num
            outputs.loss = loss

        return outputs

