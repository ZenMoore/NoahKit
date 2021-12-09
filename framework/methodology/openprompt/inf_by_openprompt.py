from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer, One2oneVerbalizer, AutomaticVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.pipeline_base import PromptForGeneration
from yacs.config import CfgNode


import torch

import json

'''
also the zero-shot training setting.
'''


def get_classes(path_to_entitylist):
    # todo : 警告！ 这个只能用于 PromptForClassification 并且只能用于 MOL 的分类！ ORG 根本没办法分类！ 请见 README !
    with open(path_to_entitylist, 'r', encoding='utf-8') as f:
        data = json.load(f)
        classes = data['whitelist']['mol']
        return classes


def get_examples(paragrahs):
    # todo
    return [  # For simplicity, there's only two examples
        # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
        InputExample(
            guid=0,
            text_a="Albert Einstein was one of the greatest intellects of his time.",
        )
    ]


def get_templates(templates):
    res = []
    for template in templates:
        # todo : 要不要区分出 org/mol 以便于确定填充实体的类型？
        if '[ORG]' in template:
            template = template.replace('[ORG]', '{"mask"}')
            # template = template.replace('[ORG]', '企业{"mask"}')
        elif '[MOL]' in template:
            template = template.replace('[MOL]', '{"mask"}')
            # template = template.replace('[MOL]', '板块{"mask"}')
        else:
            template = template  # attention : we mush consider this case in fill_templates function !
        res.append('从这段话中，我们可以了解%s{"meta": "paragraph"}' % template)
    return res


def fill_templates(templates, paragraphs, path_to_model, path_to_entitylist=None, mol_tree=None):
    questions = []
    'define a task'
    # classes = get_classes(path_to_entitylist)  # for PromptForClassification configuration
    dataset = get_examples(paragraphs)

    'obtain a PLM'
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", path_to_model)

    'define templates'
    prompt_templates = []
    for each_template in get_templates(templates):
        if not '{"mask"}' in each_template:
            questions.append(each_template)
        else:
            prompt_templates.append(ManualTemplate(
                text=each_template,
                tokenizer=tokenizer,
            ))

    # 'define a verbalizer'
    verbalizer = One2oneVerbalizer(tokenizer)

    'construct a prompt model'
    prompt_models = []
    for prompt_template in prompt_templates:
        prompt_models.append(PromptForClassification(
            template=prompt_template,
            plm=plm,
            verbalizer=verbalizer
        ))  # issue : PromptForGeneration is more suitable because the verbalizer is in fact none ?. May not, because there is one2one verbalizer

    'define a dataloader'
    dataloaders = []
    for prompt_template in prompt_templates:
        dataloaders.append(PromptDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            template=prompt_template,
            tokenizer_wrapper_class=WrapperClass,
        ))

    'inference'
    for i in range(prompt_templates):
        prompt_model = prompt_models[i]
        prompt_model.eval()
        with torch.no_grad():
            for batch in dataloaders[i]:
                logits = prompt_model(batch)
                preds = torch.argmax(logits, dim=-1)
                print(verbalizer.label_words[preds])


def fill_template(template, paragraph, path_to_model, path_to_entitylist=None, mol_tree=None):
    pass
    'define a task'
    example = InputExample(
        guid=0,
        text_a=paragraph,
    )
    dataset = [example]

    'obtain a PLM'
    with open(path_to_model) as f:
        config = CfgNode.load_cfg(f)
        config.freeze()
        plm, tokenizer, model_config = load_plm(config)

    'define template'
    prompt_template = ManualTemplate(
        text=[template],
        tokenizer=tokenizer,
    )

    # 'define a verbalizer'
    # verbalizer = One2oneVerbalizer(tokenizer)
    verbalizer = AutomaticVerbalizer(tokenizer)

    'construct a prompt model'
    prompt_model = PromptForClassification(
        template=prompt_template,
        model=plm,
        verbalizer=verbalizer
    )

    'define a dataloader'
    dataloader = PromptDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        template=prompt_template,
        batch_size=1
    )

    'inference'
    prompt_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            print(prompt_model(batch))
            # preds = torch.argmax(logits, dim=-1)
            # print(verbalizer.label_words[preds])


if __name__ == '__main__':
    template = '从这段话中，我们可以了解{"mask"}的发展情况怎么样？{"meta": "paragraph"}'
    paragraph = '英特格公司是一个全球性的开发商、制造商和供应商。该公司主要为半导体和其他高科技产业提供加工产品和材料。其产品和材料被用来制造' \
                '平板显示器、发光二极管、光致抗蚀剂、高纯化学品、燃料电池、太阳能电池、气体激光器、光学储存装置、光纤电缆和航空航天部件、生物医学应用等 '
    path_to_model = '/nfs/users/wangzekun/repos/fewqa/prompt/quesgen/model.yaml'
    fill_template(template, paragraph, path_to_model)
