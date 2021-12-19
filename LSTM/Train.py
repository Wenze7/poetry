import time
import torch
from torch import nn
import numpy as np
import os


def train(args, model, optimizer, scheduler, Criterion, train_loader):
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
            logits = model(input_ids)
            labels = input_ids

            if args.mode == 'CCPC':
                token_type_ids = sample['tokens'].to(args.device)
                mask = (token_type_ids == args.content_id).long()
                labels = labels * mask

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.permute(0, 2, 1)

            loss = Criterion(shift_logits, shift_labels)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - start_time),
                         float(scheduler.get_last_lr()[0])))
        if (epoch+1) % 2 == 0 or epoch == 0:
            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
            torch.save(model.state_dict(), output_dir + 'model_epoch{}/model'.format(epoch + 1))
            print('epoch {} finished'.format(epoch + 1))
            epoch_end_time = time.time()
            print('time for one epoch: {}'.format(epoch_end_time - epoch_start_time))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    torch.save(model.state_dict(), output_dir + 'final_model/model')
