# Hugging Face
- we note transformers only,\
because the pre-trained models, etc, are easy to use.
- we note inference only,\
because the training code is based on pytorch, which is very common.

## Project Component
- Config : class or `config.json` 
- Tokenizer : vocab.txt and `special_tokens_map.json` and `tokenizer_config.json`
- Model : when export model, we get `config.json` and `pytorch_model.bin`
