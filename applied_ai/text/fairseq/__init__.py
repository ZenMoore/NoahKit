import fairseq
import sys
from fairseq import tasks
from fairseq import criterions
from fairseq import models
from fairseq import optim
from fairseq.optim import lr_scheduler
from fairseq.models.fconv import FConvEncoder, FConvDecoder, FConvModel
from torch.utils.data import Dataset
from fairseq.data import LanguagePairDataset

'''
learning material : https://fairseq.readthedocs.io/en/latest/index.html
api : https://fairseq.readthedocs.io/en/latest/index.html
'''

args = sys.get_args()
dictionary = dict()  # vocab.txt, load to dictionary

'task'
# we arrive it from task, and then, if needed, we modify task/model/criterion/optimizer/etc...

# set up tasko
task_self = tasks.setup_task(args)
task_translation = tasks.translation.TranslationTask(args, src_dict= dictionary, tgt_dict= dictionary)

# build model and criterion
model = task_translation.build_model(args)
criterion = task_translation.build_criterion(args)

# load datasets
task_translation.load_dataset('train')
task_translation.load_dataset('valid')

# batch iteration
batch_itr = task_translation.get_batch_iterator(task_translation.dataset('train'), max_tokens= 4096)
for batch in batch_itr:
    loss, sample_size, logging_output = task_translation.get_loss(model, criterion, batch)
    loss.backward()


'model'
# we should import the model class first
# detailed construction
fconven = FConvEncoder(dictionary)  # other parameters are as default
fconvde = FConvDecoder(dictionary)  # other parameters are as default
fconvmodel = FConvModel(fconven, fconvde)
# one line construction
fconvmodel = FConvModel.build_model(args, task_translation)

'criterion'
# detailed construction
criterion = criterions.FairseqCriterion(task_translation)
loss = criterion(model, batch)
# one line construction
criterions.FairseqCriterion.build_criterion(args, task_translation)

'optimization'
# detailed construction
optimizer = optim.FP16Optimizer(args, fconvmodel.get_parameter())
# one line construction
optim.FP16Optimizer.build_optimizer(args, fconvmodel.get_parameter())

'learning rate scheduler'
lr_scheduler = lr_scheduler.FairseqLRScheduler(args, optimizer)

'dataset'
dataset = LanguagePairDataset([], src_sizes= [1, 2, 3], src_dict=dictionary)










