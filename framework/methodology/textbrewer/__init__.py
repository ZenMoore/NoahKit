import torch
import textbrewer
from transformers import BertForSequenceClassification, BertTokenizerFast, DistilBertForSequenceClassification
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
from datasets import load_dataset

'''
learning material 1 simple pipeline : from book 《自然语言处理——基于预训练模型的方法》
learning material 2 another simple pipeline : https://textbrewer.readthedocs.io/en/latest/Tutorial.html
api : https://textbrewer.readthedocs.io/en/latest/
'''


dataset = load_dataset('glue', 'sst2', split='train')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


'load dataset and construct dataloader'
def encode(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length')


dataset = dataset.map(encode, batched=True)
encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)  # todo
columns = ['input_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns)  # todo


def collate_fn(examples):
    return dict(tokenizer.pad(examples, return_tensors='pt'))  # todo


dataloader = torch.util.Dataloader(encoded_dataset, collate_fn=collate_fn, batch_size=8)


'define teacher model and student model'
teacher_model = BertForSequenceClassification.from_pretrained('bert-base-cased')
student_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')
print("teacher model's parameters:")
result, _ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
print(result)
print("student model's parameters:")
result, _ = textbrewer.utils.display_parameters(student_model, max_level=3)
print(result)


'define optimizer'
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    teacher_model.to(device)
    student_model.to(device)


'configuration'
train_config = TrainingConfig(device=device)
distill_config = DistillationConfig()


'define adaptor'
def simple_adaptor(batch, model_outputs):
    return {'logits': model_outputs[1]}


'define distiller'
distiller = GeneralDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model,
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


'launch distillation !'
with distiller:
    distiller.train(optimizer, dataloader,
                    scheduler_class=None, scheduler_args= None,
                    num_epochs=1, callbacks=None)





