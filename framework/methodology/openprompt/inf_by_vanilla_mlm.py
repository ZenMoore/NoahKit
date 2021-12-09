from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from transformers import pipeline
import torch

def run(texts, model):
    path_to_model = '/nfs/users/wangzekun/repos/fewqa/assets/models/%s' % model

    if model == 'mengzi-t5-base':
        # todo : 目前这个模型使用了 top-k 以及 output_scores, 但是一点用都没有，之后重试一下
        for i, text in enumerate(texts):
            texts[i] = text.replace('<mask>', '<extra_id_0>')

        print('loading tokenizer...')
        tokenizer = T5Tokenizer.from_pretrained(path_to_model)
        # model = T5Model.from_pretrained(path_to_model)
        # model_config = T5Config.from_pretrained(path_to_model)

        print('loading model...')
        model = T5ForConditionalGeneration.from_pretrained(path_to_model)

        # print('loading pipeline...')
        # gener = pipeline('text2text-generation', model=model, tokenizer=tokenizer, config=model.config)
        #
        # print('generating...')
        # res = gener(texts)
        # print(res)
        
        print('predicting...')
        model.eval()
        input_ids = tokenizer(texts, return_tensors='pt', padding=True).input_ids
        outputs = model.generate(input_ids, top_k=10, output_scores=True)
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(res)
        # for output in outputs:
        #     res = tokenizer.decode(output, skip_special_tokens=True)
        #     print(res)
        #     print(type(res))
        # print(outputs)

    elif model == 'bart-base-chinese':
        # todo : 目前这个模型使用的是 pipeline 实现，看看能不能使用非 Pipeline 实现以便于提供 output_scores 或者 top-k
        for i, text in enumerate(texts):
            texts[i] = text.replace('<mask>', '[MASK]')

        print('loading tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(path_to_model)
        print('loading model...')
        model = BartForConditionalGeneration.from_pretrained(path_to_model)
        print('loading pipeline...')
        text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
        print('generating...')
        res = text2text_generator(texts, max_length=99999, do_sample=False)
        print(res)
    else:
        raise NotImplementedError

    print('done !')

if __name__ == '__main__':
    template1 = '从这段话中，我们可以了解<mask>的发展情况怎么样？'
    paragraph1 = '英格达公司是一个全球性的开发商、制造商和供应商。该公司主要为半导体和其他高科技产业提供加工产品和材料。其产品和材料被用来制造' \
                '平板显示器、发光二极管、光致抗蚀剂、高纯化学品、燃料电池、太阳能电池、气体激光器、光学储存装置、光纤电缆和航空航天部件、生物医学应用等。'

    template2 = '从这段话中，我们可以了解<mask>的发展布局是什么？'
    paragraph2 = '宽频、电力管理集成电路和标准半导体 IDM 公司，产品被用于汽车，通信，计算机，消费，工业，LED 照明，医疗，军事飞机，航空航天，智能电网等领域'

    model = 'mengzi-t5-base'
    # model = 'bart-base-chinese'

    run([template1+paragraph1, template2+paragraph2], model)
