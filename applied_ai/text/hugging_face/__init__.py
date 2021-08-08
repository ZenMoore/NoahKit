import transformers
import torch
from transformers import pipeline

'''
learning material : https://www.cnblogs.com/dongxiong/p/12763923.html
api : https://huggingface.co/transformers/
'''

'model importation'
# 1. automatic download :
# model = transformers.BertModel.from_pretrained('model-name')  # network-hungry
# 2. manually download form hugging face and :
MODEL_PATH = r'../../../../../bert-base-uncased/' # todo : difficult to download : too large !
tokenizer = transformers.BertTokenizer.from_pretrained(r'../../../../../bert-base-uncased/vocab.txt')
model_config = transformers.BertConfig.from_pretrained(MODEL_PATH)
model_config.output_hidden_states = True
model_config.output_attentions = True
bert_model = transformers.BertModel.from_pretrained(MODEL_PATH, config=model_config)

'tokenizer'
# this tokenizer is not in classic sense, because it is word2idx
# - for single sentence
tokenizer.encode('I like her.')
tokenized = tokenizer.encode_plus('i like her', 'but not you.')
# tokenized['xxx'] :
# input_ids: list[int],
# token_type_ids: list[int] if return_token_type_ids is True (default)
# attention_mask: list[int] if return_attention_mask is True (default)
# overflowing_tokens: list[int] if a max_length is specified and 		return_overflowing_tokens is True
# num_truncated_tokens: int if a max_length is specified and return_overflowing_tokens is True
# special_tokens_mask: list[int] if add_special_tokens if set to True and return_special_tokens_mask is True

'inference'
# add an additive dimension : batch_size
input_ids = torch.tensor([tokenized['input_ids']])
token_type_ids = torch.tensor([tokenized['token_type_ids']])
bert_model.eval()  # convert to eval mode : not train, but evaluation
device = 'cuda'
token_tensor = input_ids.to(device)
segment_tensor = token_type_ids.to(device)

with torch.no_grad():
    outputs = bert_model(token_tensor, token_type_ids=segment_tensor)
    encoded_layers = outputs
# outputs (tuple):
# sequence_output, pooled_output, (hidden_states), (attentions)

'task'
# question answering : question + reference -> answer (copy mechanism)
from transformers import BertForQuestionAnswering as QABert
model_config.num_labels = 2  # resp. beginning and ending
model = QABert(model_config)
model.bert = bert_model

model.eval()
question, reference = 'who was Jimmy?', 'Jimmy is my classmate.'
input_ids = tokenizer.encode(question, reference)  # encode() can process one sentence or one sentence pair.
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]  # 102 is [SEP]
# start_scores and end_scores : torch.Size([batch_size, seg_length]), the possibility that a position is the start/end.
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
all_tokens = tokenizer.convert_ids_to_tokens(input_ids)  # let's see how the input is like
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1])
print(answer)

# text classification : text -> class label
XLNET_PATH = r'path-to-xlnet-model'
# different from bert model, whose word embedding is word id (vocab.txt)
# xlnet's word embedding is not vocab ids, so we should use xlnet's special tokenizer
tokenizer = transformers.XLNetTokenizer.from_pretrained(XLNET_PATH)
model_config = transformers.XLNetConfig.from_pretrained(XLNET_PATH)
model_config.num_labels = 3
cls_model = transformers.XLNetForSequenceClassification.from_pretrained(XLNET_PATH, config=model_config)
cls_model.eval()
input_ids = tokenizer.encode_plus('i like you, what about you?')
# logits:Size([batch_size, num_labels]), if model.train, return (loss, logits, hiddens)
logits, hiddens = cls_model(input_ids)
# or just two lines :
classifier = pipeline('pipeline_name', model=cls_model, tokenizer=tokenizer)
logits, hiddens = classifier('i like you, what about you?')
