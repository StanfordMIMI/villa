import torch
import time
from tqdm import tqdm
import os


def train(
    model, loss_fn, opt, train_loader, val_loader, epochs, batch_size, checkpoint_dir
):

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(0, epochs):
        model.train()
        time_start = time.time()

        for step, sample in tqdm(enumerate(train_loader)):
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                pred = model(sample)

            loss = loss_fn(pred, sample)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            summary = (
                "\r[Epoch {}][Step {}/{}] Loss: {}, Lr: {} - {:.2f} m remaining".format(
                    epoch + 1,
                    step,
                    int(len(train_loader.dataset) / batch_size),
                    "{}: {:.2f}".format(
                        type(loss_fn).__name__, loss_fn.mean_running_loss
                    )
                    * [group["lr"] for group in opt.param_groups],
                    ((time.time() - time_start) / (step + 1))
                    * ((len(train_loader.dataset) / batch_size) - step)
                    / 60,
                )
            )
            print(summary)
        time_end = time.time()
        elapse_time = time_end - time_start
        print("Finished in {}s".format(int(elapse_time)))

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"last.pkl"))
