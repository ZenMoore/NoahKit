# TextBrewer : NLP Knowledge Distillation Toolkit by HIT

## Features
1. widely applicable: multiple models and multiple tasks.
2. flexible configuration
3. multiple distillation methods and strategies.
4. easy to use.

## Architecture
- Configurations: DistillationConfig, TrainingConfig, Presets, Losses, Schedulers...  
- Distillers
    - BasicDistiller : fundamental.
    - MultiTasksDistiller: multi-teacher multi-tasks.
    - BasicTrainer: supervised training of teacher models.
    - GeneralDistiller: provide intermediate loss functions.
    - MultiTeacherDistiller: multi-teacher single task.
- Utilities: auxiliary.

## Pipeline
1. preparation:
    -  train teacher model in a supervised way by BasicTrainer.
    - define and initialize student model : random initialization or pre-trained model initialization.
    - construct dataload, optimizer and learning rate scheduler for student model.
2. define related configurations by TrainingConfig and DistillationConfig, initialize Distiller.
3. define adaptor and callback
4. Distiller.train()
- adaptor: Translate models into Distiller, which takes model input and output as input, and returns a dict like this: {'logits':..., 'hiddens':...'}
- callback: In training mode, we evaluate the model at regular steps on eval datasets. We can do it by callback. The distiller will perform callback at each checkpoint and save the distiller. 
