This package is used for question template filling with org/mol entities.

# Inference by OpenPrompt

prompt template : 

- [x] `从这段话中，我们可以了解{'meta' : 'question_template'}{'meta' : 'paragraph'}` 把 slot 换为 `{'mask'}`
- [ ] `{'meta' : 'question_template_as_statement'}    {'meta' : 'paragraph'}` 把 slot 换为 `{'mask'}`

verbalizer :

- [x] 空
- [ ] 概念映射到细分行业，细分行业映射到行业大类

> verbalizer 也可以用于对 slot 填充结果进行知识库过滤，例如 : 把非 ORG/MOL 实体名映射到弃用符号，最后凡是带有弃用符号的问题都抛弃

plm :

- [x] `langboat/mengzi-bert-base-fin` : 这个模型不能用！！！只能生成一个字！！！
- [ ] `langboat/mengzi-t5-base`
- [ ] `fnlp/bart-base-chinese`

training setting :

- [x] zero-shot : inference

prompt model : 
- [x] 使用 PromptForGeneration
- [ ] 使用 PromptForClassification :
    - [ ] 实现通过 NER 把 orgs 提出来，和知识库中的 mols 合并构成 classes
    - [ ] ORG 依旧通过 NER 填，但是 MOL 使用 classes 去填
    - :question: 有没有 non verbalizer 的 PromptForClassification
    
explore :
please read some prompt-based ner paper to learn how to design template/verbalizer/etc. 

# Inference by Vanilla MLM

- [x] `langboat/mengzi-t5-base`
- [x] `fnlp/bart-base-chinese`