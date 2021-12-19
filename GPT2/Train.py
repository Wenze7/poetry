import time
from torch import nn
import numpy as np
import os

def train(args, model, optimizer, scheduler, train_loader, content_id):
    output_dir = "./ModelConfig/model_" + args.mode + "/"
    model.train()
    start_time = time.time()
    log_steps = 100
    global_step = 0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print("start training epoch{}...".format(epoch))
        losses = []
        for step, sample in enumerate(train_loader):
            input_ids = sample["input_ids"].to(args.device)
            tokens = None
            if args.mode == 'CCPC':
                tokens = sample["tokens"].to(args.device)

            outputs = model.forward2(input_ids = input_ids, labels=input_ids, token_type_ids=tokens, content_id=content_id)
            loss, logits = outputs[:2]

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

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        epoch_end_time = time.time()
        print('time for one epoch: {}'.format(epoch_end_time - epoch_start_time))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')