import datetime
import os

import matplotlib
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from src.bounding_box import dataset
from src.bounding_box.model import BoardQuad
from src.bounding_box.inference import mask_to_corners
from src import consts

# This training flow saves plots to files only; enforce non-interactive backend.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _polygon_area(poly: torch.Tensor) -> float:
    """Shoelace formula for polygon area. poly shape: [N, 2]."""
    n = poly.shape[0]
    if n < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return abs(float(
        (x * torch.roll(y, -1) - torch.roll(x, -1) * y).sum()
    )) / 2.0


def _sutherland_hodgman_clip(
    subject: list[tuple[float, float]],
    clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Clip subject polygon by clip polygon using Sutherland-Hodgman algorithm."""

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersection(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = list(subject)
    for i in range(len(clip)):
        if len(output) == 0:
            return []
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        for j in range(len(input_list)):
            current = input_list[j]
            prev = input_list[j - 1]
            if inside(current, a, b):
                if not inside(prev, a, b):
                    output.append(intersection(prev, current, a, b))
                output.append(current)
            elif inside(prev, a, b):
                output.append(intersection(prev, current, a, b))
    return output


def _quad_iou(pred_corners: torch.Tensor, target_corners: torch.Tensor) -> float:
    """Compute IoU between two quads. Each is [4, 2] in pixel coords."""
    pred_list = [(pred_corners[i, 0].item(), pred_corners[i, 1].item()) for i in range(4)]
    target_list = [(target_corners[i, 0].item(), target_corners[i, 1].item()) for i in range(4)]

    clipped = _sutherland_hodgman_clip(pred_list, target_list)
    if len(clipped) < 3:
        return 0.0

    inter_poly = torch.tensor(clipped, dtype=torch.float32)
    inter_area = _polygon_area(inter_poly)

    pred_poly = torch.tensor(pred_list, dtype=torch.float32)
    target_poly = torch.tensor(target_list, dtype=torch.float32)
    pred_area = _polygon_area(pred_poly)
    target_area = _polygon_area(target_poly)

    union_area = pred_area + target_area - inter_area
    if union_area < 1e-8:
        return 0.0
    return inter_area / union_area


def get_quad_metrics(test_data, model):
    """Compute mean L2 corner error (pixels) and mean quad IoU on test data."""
    model.eval()
    model.to(device)
    total_l2 = 0.0
    total_iou = 0.0
    num_samples = 0
    img_size = consts.BBOX_IMAGE_SIZE

    with torch.no_grad():
        for img, corners_flat, _mask in test_data:
            logits = model(img.unsqueeze(0).to(device))  # [1, 1, H, W]
            pred_mask = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu()
            pred_corners = mask_to_corners(pred_mask)

            target_px = corners_flat.reshape(4, 2) * img_size

            if pred_corners is None:
                total_l2 += img_size  # penalty
                # iou stays 0
            else:
                l2 = (pred_corners - target_px).norm(dim=1).mean().item()
                total_l2 += l2
                total_iou += _quad_iou(pred_corners, target_px)

            num_samples += 1

    return total_l2 / num_samples, total_iou / num_samples


LOSS_REPORT_FREQ = 50
TEST_ACC_FREQ = 2000


def train(
    game: str,
    outdir="models",
    total_steps=150_000,
    batch_size=16,
    max_lr=0.0003,
    test_set_size=2000,
):
    start_time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(start_time_string)

    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    train_set = dataset.GenerativeBboxDataset(
        game=game,
        augment_ratio=0.4,
    )
    test_set = dataset.generate_fixed_test_set(game=game, size=test_set_size)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, drop_last=True,
        num_workers=16, persistent_workers=True,
        collate_fn=dataset.collate_fn,
    )

    model = BoardQuad()
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=max_lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
    )

    test_iou_list = []
    test_l2_list = []
    best_iou = -1.0
    best_model = None
    num_steps = 0
    progress = tqdm(total=total_steps, desc="Training", dynamic_ncols=True, leave=True)

    for images, _corners, masks in train_loader:
        model.train()
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)  # [B, 1, H, W]
        loss = F.binary_cross_entropy_with_logits(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        num_steps += 1
        progress.update(1)
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.5f}",
        )

        if num_steps % LOSS_REPORT_FREQ == 0:
            tqdm.write(
                f"[{num_steps}/{total_steps}] "
                f"loss: {loss.item():.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.5f}"
            )

        if num_steps % TEST_ACC_FREQ == 0 or num_steps >= total_steps:
            l2_err, quad_iou = get_quad_metrics(test_set, model)
            test_iou_list.append(quad_iou)
            test_l2_list.append(l2_err)
            tqdm.write(
                f"Num steps: {num_steps}, "
                f"Quad IoU: {quad_iou:.3f}, "
                f"L2 error: {l2_err:.1f}px"
            )

            if quad_iou > best_iou:
                best_iou = quad_iou
                best_model = model.state_dict()
                tqdm.write(f"Best model updated: Quad IoU: {best_iou:.3f}")

        if num_steps >= total_steps:
            break
    progress.close()

    game_outdir = os.path.join(outdir, game)
    os.makedirs(game_outdir, exist_ok=True)
    file_name = f"{game_outdir}/best_model_quad_{best_iou:.3f}_{start_time_string}.pth"
    print("Saving to", file_name)
    torch.save(best_model, file_name)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_iou_list, label="Quad IoU")
    plt.xlabel("Eval Step")
    plt.ylabel("IoU")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_l2_list, label="L2 Error (px)")
    plt.xlabel("Eval Step")
    plt.ylabel("Pixels")
    plt.legend()
    plt.savefig(file_name + ".png", dpi=250)
