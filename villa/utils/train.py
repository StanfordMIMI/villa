import torch
import time
import os
import numpy as np
from rich import print


def train(
    model,
    loss_fn,
    opt,
    scheduler,
    train_loader,
    val_loader,
    epochs,
    batch_size,
    checkpoint_dir,
    early_stop=False,
):
    """
    Training loop.

    Parameters:
        model (torch.nn.Module): Model
        loss_fn (torch.nn.Module): Loss function
        opt (torch.optim): Optimizer
        scheduler (torch.optim.lr_scheduler): LR scheduler (set to None if no scheduler)
        train_loader (torch.utils.data.DataLoader): Dataloader for training data
        val_loader (torch.utils.data.DataLoader): Dataloader for validation data
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        checkpoint_dir (str): Directory for storing model weights
        early_stop (bool): True if early stopping based on validation loss
    """
    scaler = torch.cuda.amp.GradScaler()
    epochs_no_improvements = 0
    best_val_loss = np.inf

    for epoch in range(0, epochs):
        model.train()
        time_start = time.time()

        for step, sample in enumerate(train_loader):
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
                    ),
                    *[group["lr"] for group in opt.param_groups],
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

        if early_stop:
            val_loss = evaluate(model, loss_fn, val_loader)
            epochs_no_improvements += 1
            if val_loss < best_val_loss:
                print("Saving best model")
                torch.save(
                    model.state_dict(), os.path.join(checkpoint_dir, f"best.pkl")
                )
                epochs_no_improvements = 0
                best_val_loss = val_loss

            if scheduler:
                scheduler.step(val_loss)

            if epochs_no_improvements == 5:
                print("Early stop reached")
                return


def evaluate(model, loss_fn, val_loader, split="val"):
    """
    Validation loop.

    Parameters:
        model (torch.nn.Module): Model
        loss_fn (torch.nn.Module): Loss function
        val_loader (torch.utils.data.DataLoader): Dataloader for validation data
        split (str): Evaluation split
    Returns:
        loss (torch.Tensor): Validation loss
    """
    print(f"Evaluating on {split}")
    model.eval()

    running_loss = 0
    num_batches = 0
    with torch.no_grad():
        for step, sample in enumerate(val_loader):
            pred = model(sample)

            running_loss += loss_fn(pred, sample)
            num_batches += 1

    loss = running_loss / num_batches
    print(f"Loss = {loss}")
    return loss
