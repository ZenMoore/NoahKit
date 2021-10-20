# NoahKit
Code notebook for python package learning.

## Checkbar

| emoji                     | connotation   | tag                      |
| ------------------------- | ------------- | ------------------------ |
| 🌕 or :white_check_mark:   | experienced   | `:full_moon:`            |
| 🌔 or :white_check_mark:   | advanced      | `:moon:`                 |
| 🌓 or :white_check_mark:   | well-grounded | `:first_quarter_moon:`   |
| 🌒 or :white_large_square: | understanding | `:waxing_crescent_moon:` |
| 🌑 or :white_large_square: | strange       | `:new_moon:`             |

## Checklist

### ML Framework

integrated machine/deep/reinforcement learning toolkits.

##### Deep Learning and Neural Network

- [x] :first_quarter_moon: **pytorch** : open source deep learning framework commonly used in <u>academia</u>.
- [x] :first_quarter_moon: **tensorflow** : open source deep learning framework commonly used in <u>industry</u>.
- [ ] :new_moon: **paddle** : <u>baidu</u> open source deep learning platform derived from <u>industry practices</u>.
- [ ] :new_moon: **ncnn** : <u>tencent</u> high performance neural network forward computing framework optimized for <u>mobiles</u>.

##### Reinforcement Learning

- [ ] :waxing_crescent_moon: **gym** : a toolkit for developing and comparing reinforcement learning algorithms.
- [ ] :new_moon: **ACME** : a <u>research framework</u> for reinforcement learning by <u>DeepMind</u>.

##### Machine Learning

- [x] :first_quarter_moon: **sklearn** : machine learning library.

##### Methodology

- [ ] :new_moon: **autokeras** : <u>automl</u> system based on <u>keras</u>. 
- [ ] :new_moon: **pytorch_geometric** : <u>geometric deep learning</u> extension library for pytorch.
- [x] :first_quarter_moon: **textbrewer**: nlp <u>knowledge distillation</u> toolkit.
- [ ] :new_moon: **OpenPrompt** : <u>prompt tuning</u> toolkit developed by Tsinghua.
- [ ] :new_moon: **OpenAttack** : an open-source package for <u>texual adversarial attack</u>.
- [ ] :new_moon: **PyCID**: a python library for <u>causal influence</u> diagrams.

### Applied AI

NLP/CV/etc. packages.

##### Vision

- [x] :moon: **torchvision** : vision processing toolkit in <u>pytorch</u>.
- [x] :first_quarter_moon: ​**cv2** : <u>opencv</u> in python.
- [x] :first_quarter_moon: ​**pillow** : python imaging library.

##### Text

- [x] :first_quarter_moon: **torchtext** : natural language data processing for <u>pytorch</u>.

- [x] :first_quarter_moon: **nltk** : <u>english</u> natural language toolkit by <u>UPenn</u>.

  `for chinese, we should import Stanford NLP Chinese package.`

- [x] :first_quarter_moon: **ltp** : <u>chinese</u> language technology platform by <u>HIT</u>.

- [x] :first_quarter_moon: **gensim** : <u>topic</u> modelling for humans (<u>semantic vector</u>).

- [x] :first_quarter_moon: **glove** : toy python implementation of <u>glove</u>.

- [x] :first_quarter_moon: ​**hugging face** : sota <u>pre-trained language mode</u>l toolkit.

  > mainly for pre-trained and transformer, etc.

- [x] :first_quarter_moon: ​**fairseq** : a <u>seq2seq</u> learning toolkit for <u>pytorch</u> by <u>fair</u>.

- [ ] :new_moon: **pattern** : simpler to get started than nltk.

- [ ] :new_moon: [**pyenchant**](http://link.zhihu.com/?target=https%3A//github.com/rfk/pyenchant) : easy access to dictionaries

- [ ] :waxing_crescent_moon: **allennlp** : natural language processing toolkit based on <u>pytorch</u> by <u>Allen</u>.

  > mainly for ELMo, KnowBERT, etc.

- [ ] :new_moon: **seqeval** : evaluation for sequence labelling.

- [ ] :waxing_crescent_moon: **​(corenlp, spacy, jieba)** : chinese processing, language processing, etc.

`we put gensim and glove together in this code notebook.`

##### Audio

- [ ] :new_moon: ​**torchaudio** : audio processing toolkit in <u>pytorch</u>.

### Data Science

data processing and computing.

##### Processing

- [x] :first_quarter_moon: **pandas** : high performance and user-friendly <u>data structure and analysis</u> tool.
- [x] :moon: **matplotlib** : comprehensive library for creating <u>static, animated and interactive</u> <u>visualizations</u>  in python.

##### Computing

- [x] :first_quarter_moon: **numpy** : <u>scientific computing</u>.
- [x] :first_quarter_moon: **scipy** : open-source software for <u>mathematics, science, and engineering</u>.

### DevTool

programming aids.

##### Running Auxiliary

- [ ] :waxing_crescent_moon: ​**tqdm** : <u>progress bar</u>.
- [ ] :new_moon: **labml** : <u>training visualization</u> on the <u>mobile phone</u>.
- [ ] :new_moon: **wandb**: <u>experiment results</u> management.
- [ ] :new_moon: **TensorSensor**：<u>visualization of errors</u> about tensor.
- [ ] :new_moon: **optuna**: automatic <u>hyperparameter</u> searching.
- [ ] :new_moon: **alfred**: visualization of <u>annotations</u>.

##### Optimization

- [x] :first_quarter_moon: **threading** : <u>multithreading</u>.
- [x] :first_quarter_moon: **multiprocessing** : <u>multiprocessing</u>.
- [ ] :new_moon: **deepspeed** : microsoft <u>training acceleration</u> toolkit.

`We put threading and multiprocessing together in this code notebook.`

##### File System

- [ ] :new_moon: ​**glob** : <u>file path</u> manager.
- [ ] :new_moon: ​**shutil** : advanced <u>file operation</u> module.

##### Information Retrieval

- [ ] :new_moon: ​**re** : <u>regex</u>.
- [ ] :new_moon: ​**elasticsearch** : <u>search and analysis engine</u>.
- [x] :first_quarter_moon: **scrapy** : <u>web crawler</u>.

##### Towards User

- [ ] :new_moon: ​**pyqt** : <u>GUI</u>.    
- [ ] :new_moon: ​**pyinstaller** : <u>application packaging</u>.

##### Operating System

- [ ] :waxing_crescent_moon: **os** : access <u>os</u>-related functions.
- [ ] :new_moon: ​**sys** : access <u>interpreter</u>-related variables and functions.

##### Encoding Management

- [ ] :new_moon: [**unidecode**](http://link.zhihu.com/?target=https%3A//pypi.python.org/pypi/Unidecode/) : because ascii is much easier to deal with
- [ ] :new_moon: [**chardet**](http://link.zhihu.com/?target=https%3A//pypi.python.org/pypi/chardet) : character encoding detection

## Note
- .py file is not runnable : we view it as a codebook composed by numerous fragments, but not an executable file 
(there is no strong relation between every two fragments).
