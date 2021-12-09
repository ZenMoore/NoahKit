import deepspeed
import argparse
import torch
import json

'''
learning material 1 deepspeed : https://www.deepspeed.ai/
learning material 2 huggingface : https://huggingface.co/docs/transformers/master/en/main_classes/deepspeed
api : https://deepspeed.readthedocs.io/en/latest/
'''

'distributed environment setup'
deepspeed.init_distributed('nccl')

'initializing'
args = argparse.ArgumentParser().parse_args()
model = None
model_params = None
model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(args, model, model_parameters=model_params,
                                                                         optimizer=torch.optim.Adam)
# if learning rate scheduler is supposed to be executed at every training step, use this initialize() function.
# but if learning rate scheduler is supposed to be executed at any other interval (like epoches), please define it explicitly.

'training'

for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()

'checkpointing'


def dataloader_to_step(dataloader, step):
    for i, batch in enumerate(dataloader):
        if i == step - 1:
            return


_, client_sd = model_engine.load_checkpoint(args.load_dir,
                                            args.ckpt_id)  # ckpt_id is to identify checkpoint in the dir, can be time/date/loss/etc.
step = client_sd['step']

# advance data loader to ckpt step
dataloader_to_step(dataloader, step + 1)
for step, batch in enumerate(dataloader):

    # forward() method
    loss = model_engine(batch)

    # runs backpropagation
    model_engine.backward(loss)

    # weight update
    model_engine.step()

    # save checkpoint
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)

'commandline'
# in the command line, we can use deepspeed_config.json
# please see applied_ai/text/hugging_face for a model definition
# that is almost the same, except that we need to get optimizer/lr_scheduler/etc. from deepspeed (see ./deepspeed.py)
# so this kind of usage is focused on the shell and ds_config.json

path_to_ds_config = './ds_config.json'
with open(path_to_ds_config, 'r', encoding='utf8') as f:
    config = json.load(f)
    model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(args, model, model_parameters=model_params,
                                                                             optimizer=torch.optim.Adam, config_params=config)

