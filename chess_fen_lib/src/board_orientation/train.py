# Import modules
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from src.board_orientation import dataset
from src.board_orientation.model import OrientationModel
from src.consts import get_training_pgn_files
import os
import datetime

device = torch.device("cpu")


def get_accuracy_and_loss(loader, model, criterion):
    num_correct = 0
    num_samples = 0
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for input, target in loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            loss += criterion(output, target).item() * target.size(0)

            output = output.cpu()
            target = target.cpu()

            num_correct += torch.count_nonzero(torch.abs(output - target) < 0.5)
            num_samples += target.size(0)
    model.train()
    return num_correct / num_samples, loss / num_samples


LOSS_REPORT_FREQ = 100
TEST_ACC_FREQ = 2000


def train(
    pgn_dir="resources/pychess_games",
    game: str = "chess",
    outdir="models",
    total_steps=20000,
    batch_size=8,
    max_lr=0.001,
    train_test_split=0.7,
    rotate_probability=0.3,  # we don't rotate half of the time, since in reality, the majority of chess diagrams will be depicted from the white side
):
    start_time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(start_time_string)

    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    all_files = get_training_pgn_files(pgn_dir)
    split_idx = max(1, int(len(all_files) * train_test_split))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:] if len(all_files) > split_idx else all_files

    max_train = total_steps * batch_size
    max_test = int(max_train * (1.0 - train_test_split) / train_test_split)

    train_set = dataset.BoardOrientationDataset(
        pgn_files=train_files,
        game=game,
        rotate_probability=rotate_probability,
        max=max_train,
    )
    test_set = dataset.BoardOrientationDataset(
        pgn_files=test_files,
        game=game,
        rotate_probability=rotate_probability,
        max=max_test,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = OrientationModel(game=game)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps
    )

    test_loss_list = []
    test_acc_list = []
    best_acc = -1.0
    best_model = None
    num_steps = 0
    progress = tqdm(total=total_steps, desc="Training", dynamic_ncols=True, leave=True)

    while num_steps < total_steps:
        running_loss = 0.0

        for i, (input, target) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            num_steps += 1
            progress.update(1)
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.5f}",
            )

            running_loss += loss.item()

            if (i + 1) % LOSS_REPORT_FREQ == 0:
                tqdm.write(
                    f"[{num_steps}/{total_steps}, {i + 1:5d}] "
                    f"loss: {running_loss / LOSS_REPORT_FREQ:.4f}, "
                    f"lr: {optimizer.param_groups[0]['lr']:.5f}"
                )
                running_loss = 0.0

            if (i + 1) % TEST_ACC_FREQ == 0 or num_steps >= total_steps:
                test_acc, test_loss = get_accuracy_and_loss(
                    test_loader, model, criterion
                )
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                tqdm.write(
                    f"Num steps: {num_steps}, "
                    f"Test Loss: {test_loss_list[-1]:.4f}, "
                    f"Test Acc: {test_acc_list[-1]:.3f}"
                )

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = model.state_dict()
                    tqdm.write(f"Best model updated: Test Acc: {best_acc:.3f}")

            if num_steps >= total_steps:
                break
    progress.close()

    game_outdir = os.path.join(outdir, game)
    os.makedirs(game_outdir, exist_ok=True)
    file_name = f"{game_outdir}/best_model_orientation_{best_acc:.3f}_{start_time_string}.pth"
    print("Saving to", file_name)
    torch.save(best_model, file_name)

    # Plot the loss and accuracy curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_loss_list, label="Test Loss")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_acc_list, label="Test Acc")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(file_name + ".png", dpi=250)
