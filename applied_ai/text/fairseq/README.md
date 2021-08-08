# Fairseq

## Command Line
we can use terminal and shell script (.sh).

**download pretrained model**
```shell script
> curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf
```
**generate translation**
```shell script
> MODEL_DIR=wmt14.en-fr.fconv-py
> fairseq-interactive \
    --path $MODEL_DIR/model.pt $MODEL_DIR \
    --beam 5 --source-lang en --target-lang fr \
    --tokenizer moses \
    --bpe subword_nmt --bpe-codes $MODEL_DIR/bpecodes
```
**data pre-processing**
```shell script
> cd examples/translation/
> bash prepare-iwslt14.sh
> cd ../..
> TEXT=examples/translation/iwslt14.tokenized.de-en
> fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en
```
**training**
```shell script
> mkdir -p checkpoints/fconv
> CUDA_VISIBLE_DEVICES=0
> fairseq-train data-bin/iwslt14.tokenized.de-en \
    --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir checkpoints/fconv
```
**generation**
```shell script
> fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/fconv/checkpoint_best.pt \
    --batch-size 128 --beam 5
```
**distribution**
```shell script
> python -m torch.distributed.launch \
    --nproc_per_node=8\
    --nnodes=2 \ 
    --node_rank=0 --master_addr="192.168.1.1" \
    --master_port=12345 \
    $(which fairseq-train) data-bin/wmt16_en_de_bpe32k \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0005 --min-lr 1e-09 \
    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16
```
**Sharding ver large datasets**
```shell script
> fairseq-train data-bin1:data-bin2:data-bin3 (...)
```

## Example : simple LSTM
make sure you have define and register the simple LSTM model and architecture with python. \
and then : \
**training**
```shell script
> fairseq-train data-bin/iwslt14.tokenized.de-en \
  --arch tutorial_simple_lstm \
  --encoder-dropout 0.2 --decoder-dropout 0.2 \
  --optimizer adam --lr 0.005 --lr-shrink 0.5 \
  --max-tokens 12000
```
**generation**
```shell script
> fairseq-generate data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/checkpoint_best.pt \
  --beam 5 \
  --remove-bpe
```


## Example : classifier RNN
make sure you have download the dataset : https://dl.fbaipublicfiles.com/fairseq/data/tutorial_names.tar.gz 

**data preprocessing**
```shell script
> fairseq-preprocess \
  --trainpref names/train --validpref names/valid --testpref names/test \
  --source-lang input --target-lang label \
  --destdir names-bin --dataset-impl raw
```
we get dataset dirs : /names and /names-bin 

**register a new model** 
see /classifier/models/, put the model in /fairseq/models/

**register a new task** 
see /classifier/tasks, put the task in /fairseq/tasks/

**train the model**
```shell script
> fairseq-train names-bin \
  --task simple_classification \
  --arch pytorch_tutorial_rnn \
  --optimizer adam --lr 0.001 --lr-shrink 0.5 \
  --max-tokens 1000
```
we get model dirs : /checkpoints

**evaluate the model**
see /classifier/eval_classifier.py
```shell script
python eval_classifier.py names-bin --path checkpoints/checkpoint_best.pt
```



