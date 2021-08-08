import torch.hub as hub

en2de = hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('hello world', beam=5)