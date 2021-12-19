import torch
import time
import torch.nn as nn
import numpy as np
import os


def train(args, model, optimizer, scheduler, data_loader, tokenizer):
    model.train()
    start_time = time.time()
    log_steps = 100
    global_step = 0
    output_dir = "./ModelConfig/model_" + args.mode + "/"


    for epoch in range(args.epochs):
        losses = []
        print('start train epoch:{}'.format(epoch))
        for step, sample in enumerate(data_loader):

            # title = sample['input_ids'].to(args.device, dtype=torch.long)
            # label = sample['label_ids'].to(args.device, dtype=torch.long)
            # attention_mask = sample['attention_mask'].to(args.device, dtype=torch.long)
            # decoder_attention_mask = sample['decoder_attention_mask'].to(args.device, dtype=torch.long)

            title = sample['zh_input_ids'].to(args.device, dtype=torch.long)
            label = sample['en_input_ids'].to(args.device, dtype=torch.long)
            attention_mask = sample['zh_attention_mask'].to(args.device, dtype=torch.long)
            decoder_attention_mask = sample['en_attention_mask'].to(args.device, dtype=torch.long)

            outputs = model(input_ids=title, attention_mask=attention_mask,
                            labels=label, decoder_attention_mask=decoder_attention_mask)

            # print(outputs)

            loss = outputs.loss
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - start_time),
                         float(scheduler.get_last_lr()[0])))
                #generate('巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。', model, tokenizer)
        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))



    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')


def generate(title, model, tokenizer):
    title = tokenizer.encode_plus(title,
                                               add_special_tokens=True,
                                               max_length=10,
                                               return_token_type_ids=True,
                                               pad_to_max_length=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')

    sample_outputs = model.generate(
        input_ids=title['input_ids'].to('cuda:1', dtype=torch.long),
        attention_mask=title['attention_mask'].to('cuda:1', dtype=torch.long),
        max_length=64,
        num_beams=3,
        top_p=0.95,
        top_k=50,
        repetition_penalty=10.0,
        length_penalty=1.0,
        do_sample=True,
        early_stopping=True,
        num_return_sequences=3
    )

    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))


