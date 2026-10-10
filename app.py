"""
Deep Learning Vision Lab - Streamlit App (PyTorch, CPU friendly)
----------------------------------------------------------------
1. Small CNN for image classification (configurable) + live explanation of
   convolution, filters, stride, padding, activation and pooling
2. Transfer learning with a pre-trained MobileNetV2 / MobileNetV3-Small / ResNet18
3. Feature extraction vs fine-tuning under ONE consistent validation protocol
4. Object detection experiment (SSDlite / Faster R-CNN, boxes + confidence)
5. Performance and computational-requirements discussion (auto tables + text)

Run locally:  streamlit run app_dl.py
"""

import io
import os
import time
import urllib.request
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_digits, load_sample_images
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torchvision import models

st.set_page_config(page_title="Deep Learning Vision Lab", page_icon="🧠", layout="wide")
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

SEED = 42
BASE = 64  # all images are stored at 64x64 RGB and resized on demand
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DET_SSD = "SSDlite320 MobileNetV3-Large (fast)"
DET_FRCNN = "Faster R-CNN MobileNetV3-Large 320 FPN (more accurate)"
TL_MODELS = ["MobileNetV2", "MobileNetV3-Small", "ResNet18"]
MAX_UNFREEZE = {"MobileNetV2": 8, "MobileNetV3-Small": 6, "ResNet18": 4}
DEFAULT_UNFREEZE = {"MobileNetV2": 3, "MobileNetV3-Small": 3, "ResNet18": 1}


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_digits_arrays():
    d = load_digits()
    imgs = []
    for im in d.images:
        pil = Image.fromarray((im / 16.0 * 255).astype(np.uint8))
        pil = pil.resize((BASE, BASE), Image.Resampling.BICUBIC).convert("RGB")
        imgs.append(np.asarray(pil))
    return np.stack(imgs), np.array([str(t) for t in d.target])


@st.cache_data(show_spinner=False)
def load_zip_arrays(zip_bytes):
    xs, ys = [], []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if name.endswith("/") or "__MACOSX" in name or base.startswith("."):
                continue
            if not name.lower().endswith(IMG_EXT):
                continue
            parts = [p for p in name.split("/") if p]
            if len(parts) < 2:
                continue
            try:
                img = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                img = img.resize((BASE, BASE), Image.Resampling.BILINEAR)
            except Exception:  # noqa: BLE001
                continue
            xs.append(np.asarray(img))
            ys.append(parts[-2])
    if not xs:
        return np.zeros((0, BASE, BASE, 3), np.uint8), np.array([])
    return np.stack(xs), np.array(ys)


def sidebar_data():
    """Shared dataset + fixed stratified 60/20/20 split used by every experiment."""
    st.sidebar.header("Dataset (shared by all tabs)")
    src = st.sidebar.radio("Source", ["Built-in digits (demo)", "Upload ZIP (folder per class)"])
    if src.startswith("Built-in"):
        X, y = load_digits_arrays()
        st.sidebar.caption("scikit-learn digits: 1797 images, 10 classes.")
    else:
        zf = st.sidebar.file_uploader("ZIP file", type=["zip"])
        if zf is None:
            st.sidebar.info("Upload a ZIP: `cats/1.jpg`, `dogs/2.jpg`, ...")
            return None
        X, y = load_zip_arrays(zf.getvalue())
        if len(X) == 0:
            st.sidebar.error("No images found in the ZIP.")
            return None

    classes = sorted(np.unique(y).tolist())
    if len(classes) < 2:
        st.sidebar.error("At least 2 classes are required.")
        return None
    cidx = {c: i for i, c in enumerate(classes)}
    yi = np.array([cidx[c] for c in y])
    if np.bincount(yi).min() < 5:
        st.sidebar.error("Every class needs at least 5 images.")
        return None

    max_n = st.sidebar.slider("Max images used (speed control)", 200, 1800, 800, step=100)
    idx = np.arange(len(X))
    try:
        if max_n < len(X):
            idx, _ = train_test_split(idx, train_size=max_n, stratify=yi, random_state=SEED)
        tr, tmp = train_test_split(idx, test_size=0.4, stratify=yi[idx], random_state=SEED)
        va, te = train_test_split(tmp, test_size=0.5, stratify=yi[tmp], random_state=SEED)
    except ValueError as exc:
        st.sidebar.error(f"Split failed: {exc}")
        return None

    st.sidebar.write(f"**Classes:** {len(classes)}")
    st.sidebar.write(f"**Train / Val / Test:** {len(tr)} / {len(va)} / {len(te)}")
    st.sidebar.caption("Same stratified split (seed 42) for every model -> fair comparison.")
    return {"X": X, "y": yi, "classes": classes, "tr": tr, "va": va, "te": te}


def to_tensor(X, size, imagenet):
    t = torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return (t - MEAN) / STD if imagenet else (t - 0.5) / 0.5


def make_tensors(data, size, imagenet):
    X, y = data["X"], data["y"]
    out = []
    for key in ("tr", "va", "te"):
        ids = data[key]
        out.append(to_tensor(X[ids], size, imagenet))
        out.append(torch.from_numpy(y[ids]).long())
    return tuple(out)  # Xtr, ytr, Xva, yva, Xte, yte


# --------------------------------------------------------------------------
# Training / evaluation utilities
# --------------------------------------------------------------------------
def count_macs(model, size):
    """Multiply-accumulate operations for one image (conv + linear layers)."""
    total = [0]
    hooks = []

    def conv_hook(m, _i, o):
        total[0] += (o.numel() // o.shape[0]) * (m.in_channels // m.groups) * \
            m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, _i, _o):
        total[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hooks:
        h.remove()
    return total[0]


def evaluate(model, X, y, bs=128):
    model.eval()
    loss_sum, preds = 0.0, []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = model(X[i:i + bs])
            loss_sum += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
            preds.append(out.argmax(1))
    preds = torch.cat(preds)
    return loss_sum / len(X), (preds == y).float().mean().item(), preds.numpy()


def run_experiment(name, model, tensors, n_classes, in_size, epochs, lr, batch, mode_fn, cb):
    """Identical training/validation/test procedure for every model."""
    xtr, ytr, xva, yva, xte, yte = tensors
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(SEED)
    hist, best_val, best_state = [], -1.0, None
    t0 = time.time()
    for ep in range(epochs):
        mode_fn(model)
        perm = torch.randperm(len(xtr), generator=gen)
        loss_sum, correct = 0.0, 0
        for i in range(0, len(xtr), batch):
            b = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            out = model(xtr[b])
            loss = F.cross_entropy(out, ytr[b])
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(b)
            correct += (out.argmax(1) == ytr[b]).sum().item()
        vl, va, _ = evaluate(model, xva, yva)
        row = {"epoch": ep + 1, "train_loss": loss_sum / len(xtr), "train_acc": correct / len(xtr),
               "val_loss": vl, "val_acc": va}
        hist.append(row)
        if va > best_val:
            best_val = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        cb(ep + 1, epochs, row)
    train_time = time.time() - t0

    model.load_state_dict(best_state)  # best epoch by VALIDATION accuracy
    t1 = time.time()
    _, test_acc, preds = evaluate(model, xte, yte)
    infer_ms = (time.time() - t1) / len(xte) * 1000
    yt = yte.numpy()
    p, r, f1, _ = precision_recall_fscore_support(yt, preds, average="macro", zero_division=0)
    total = sum(p_.numel() for p_ in model.parameters())
    trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    return {
        "name": name, "history": pd.DataFrame(hist).set_index("epoch"),
        "total_params": total, "trainable_params": trainable,
        "macs": count_macs(model, in_size), "train_time": train_time,
        "infer_ms": infer_ms, "best_val": best_val, "test_acc": test_acc,
        "test_prec": p, "test_rec": r, "test_f1": f1,
        "cm": confusion_matrix(yt, preds, labels=list(range(n_classes))),
    }


def summary_row(res):
    return {
        "Model": res["name"],
        "Total params (M)": round(res["total_params"] / 1e6, 3),
        "Trainable params (M)": round(res["trainable_params"] / 1e6, 3),
        "GMACs / image": round(res["macs"] / 1e9, 4),
        "Size (MB, fp32)": round(res["total_params"] * 4 / 1e6, 1),
        "Train time (s)": round(res["train_time"], 1),
        "Inference (ms/img)": round(res["infer_ms"], 2),
        "Best val acc": round(res["best_val"], 4),
        "Test acc": round(res["test_acc"], 4),
        "Test precision": round(res["test_prec"], 4),
        "Test recall": round(res["test_rec"], 4),
        "Test F1 (macro)": round(res["test_f1"], 4),
    }


def plot_confusion(cm, classes, title):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    if len(classes) <= 12:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def show_metrics(res):
    c = st.columns(5)
    c[0].metric("Test accuracy", f"{res['test_acc']:.4f}")
    c[1].metric("Precision (macro)", f"{res['test_prec']:.4f}")
    c[2].metric("Recall (macro)", f"{res['test_rec']:.4f}")
    c[3].metric("F1 (macro)", f"{res['test_f1']:.4f}")
    c[4].metric("Train time", f"{res['train_time']:.1f}s")


def progress_ui():
    bar = st.progress(0.0)
    status = st.empty()

    def cb(ep, total, row):
        bar.progress(ep / total)
        status.write(f"Epoch {ep}/{total} | train loss {row['train_loss']:.3f} | "
                     f"val loss {row['val_loss']:.3f} | val acc {row['val_acc']:.3f}")

    return cb


def remember(res):
    st.session_state.setdefault("perf_cls", {})[res["name"]] = summary_row(res)


# --------------------------------------------------------------------------
# Tab 1: Small CNN
# --------------------------------------------------------------------------
ACTS = {"ReLU": nn.ReLU, "LeakyReLU": lambda: nn.LeakyReLU(0.1), "Tanh": nn.Tanh}


class SmallCNN(nn.Module):
    def __init__(self, n_classes, n_blocks, base, k, stride, padding, act, pool, dropout, size=32):
        super().__init__()
        pad = k // 2 if padding == "same" else 0
        layers, c_in = [], 3
        for i in range(n_blocks):
            c_out = base * (2 ** i)
            layers.append(nn.Conv2d(c_in, c_out, k, stride=stride if i == 0 else 1, padding=pad))
            layers.append(ACTS[act]())
            if pool == "Max":
                layers.append(nn.MaxPool2d(2))
            elif pool == "Average":
                layers.append(nn.AvgPool2d(2))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 3, size, size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(flat, 64), ACTS[act](),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def describe_layer(layer):
    if isinstance(layer, nn.Conv2d):
        return (f"{layer.out_channels} filters {layer.kernel_size[0]}x{layer.kernel_size[1]}, "
                f"stride {layer.stride[0]}, padding {layer.padding[0]}")
    if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
        return "window 2x2, stride 2"
    if isinstance(layer, nn.Linear):
        return f"{layer.in_features} -> {layer.out_features}"
    if isinstance(layer, nn.Dropout):
        return f"p = {layer.p}"
    return ""


def layer_table(model, size=32):
    x = torch.zeros(1, 3, size, size)
    rows = [{"Layer": "Input", "Detail": "RGB image", "Output shape": f"3x{size}x{size}", "Params": 0}]
    model.eval()
    with torch.no_grad():
        for part in (model.features, model.classifier):
            for i, layer in enumerate(part):
                x = layer(x)
                shape = "x".join(str(s) for s in x.shape[1:])
                rows.append({
                    "Layer": layer.__class__.__name__, "Detail": describe_layer(layer),
                    "Output shape": shape, "Params": sum(p.numel() for p in layer.parameters()),
                })
    return pd.DataFrame(rows)


def _fill_explain(cfg, k, s, p, first_out, n1, w1):
    act_txt = {
        "ReLU": 'ReLU = max(0, x): cheap, avoids vanishing gradients, but neurons can "die" for negative inputs.',
        "LeakyReLU": "LeakyReLU = x if x > 0 else 0.1x: keeps a small gradient for negative inputs.",
        "Tanh": "Tanh squashes values to (-1, 1): smooth, but gradients vanish for large |x| so it trains slower.",
    }[cfg["act"]]
    pool_txt = {
        "Max": "Max pooling keeps the strongest response in every 2x2 window: height and width are halved, "
               "a little translation invariance is gained and computation is reduced.",
        "Average": "Average pooling takes the mean of every 2x2 window: height and width are halved and the "
                   "maps are smoothed.",
        "None": "No pooling is used, so feature maps stay large and the fully connected layer needs many "
                "more parameters.",
    }[cfg["pool"]]
    pad_txt = ("the spatial size is preserved (for stride 1)" if cfg["padding"] == "same"
               else "every convolution shrinks the map by K-1 pixels and border pixels are seen less often")
    return f"""
**Convolution.** A small {k}x{k} window slides over the image; at every position it multiplies the
pixels by the filter weights, sums them and adds a bias. This detects local patterns (edges, strokes,
corners). The same weights are reused everywhere (weight sharing), so far fewer parameters are needed
than in a fully connected layer. Your model has **{cfg['n_blocks']} convolution block(s)**.

**Filters.** Each filter produces one feature map. Block 1 uses **{n1} filters** of size {k}x{k}x3
-> {w1:,} parameters (weights + biases) and outputs `{first_out}` (channels x height x width).
Deeper blocks double the number of filters: early layers learn simple edges, later layers combine them
into more complex shapes.

**Stride.** The step of the sliding window. The first convolution uses **stride {s}**.
Output size = floor((N + 2P - K) / S) + 1 = floor((32 + 2*{p} - {k}) / {s}) + 1.
A larger stride shrinks the feature map (cheaper, but less detail).

**Padding.** Zeros are added around the border. **'{cfg['padding']}'** padding means P = {p}:
{pad_txt}.

**Activation ({cfg['act']}).** A non-linear function applied after every convolution; without it,
stacked convolutions collapse into one linear operation. {act_txt}

**Pooling ({cfg['pool']}).** {pool_txt}
"""


def plot_filters(model):
    w = model.features[0].weight.detach().cpu().numpy()  # (C, 3, k, k)
    n = min(len(w), 16)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            f = w[i].transpose(1, 2, 0)
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f, interpolation="nearest")
    fig.tight_layout()
    return fig


def plot_feature_maps(model, x):
    model.eval()
    with torch.no_grad():
        fm = model.features[:2](x.unsqueeze(0))[0].numpy()  # conv + activation
    n = min(len(fm), 16)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(fm[i], cmap="viridis")
    fig.tight_layout()
    return fig


def small_cnn_tab(data):
    st.subheader("1. Small CNN for image classification")
    c = st.columns(4)
    n_blocks = c[0].slider("Conv blocks", 1, 3, 2)
    base = c[1].selectbox("Filters in block 1", [8, 16, 32], index=1)
    k = c[2].selectbox("Kernel size", [3, 5], index=0)
    stride = c[3].selectbox("Stride (first conv)", [1, 2], index=0)
    c = st.columns(4)
    padding = c[0].selectbox("Padding", ["same", "valid"])
    act = c[1].selectbox("Activation", list(ACTS.keys()))
    pool = c[2].selectbox("Pooling", ["Max", "Average", "None"])
    dropout = c[3].slider("Dropout", 0.0, 0.6, 0.25, step=0.05)
    c = st.columns(3)
    epochs = c[0].slider("Epochs", 1, 30, 10, key="cnn_ep")
    lr = c[1].select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3,
                            format_func=lambda v: f"{v:g}", key="cnn_lr")
    batch = c[2].selectbox("Batch size", [16, 32, 64], index=1, key="cnn_bs")

    cfg = dict(n_blocks=n_blocks, base=base, k=k, stride=stride, padding=padding, act=act,
               pool=pool, dropout=dropout)
    n_classes = len(data["classes"])
    try:
        set_seed()
        model = SmallCNN(n_classes, n_blocks, base, k, stride, padding, act, pool, dropout)
        table = layer_table(model)
    except Exception as exc:  # noqa: BLE001
        st.error("These settings shrink the image too much (feature map becomes smaller than the "
                 "kernel/pooling window). Use 'same' padding, fewer blocks or no stride/pooling. "
                 f"Details: {exc}")
        return

    with st.expander("Architecture (layer-by-layer shapes and parameters)", expanded=True):
        st.dataframe(table)
        total = sum(p.numel() for p in model.parameters())
        st.caption(f"Input: 32x32 RGB | Total parameters: {total:,} | "
                   f"Size ~ {total * 4 / 1e6:.2f} MB | ~{count_macs(model, 32) / 1e6:.2f} MMACs/image")
    with st.expander("Explanation: convolution, filters, stride, padding, activation, pooling",
                     expanded=True):
        st.markdown(_fill_explain(cfg, k, stride, k // 2 if padding == "same" else 0,
                                  table.iloc[1]["Output shape"], base,
                                  base * 3 * k * k + base))

    if st.button("Train small CNN", type="primary"):
        tensors = make_tensors(data, 32, imagenet=False)
        set_seed()
        model = SmallCNN(n_classes, n_blocks, base, k, stride, padding, act, pool, dropout)
        cb = progress_ui()
        res = run_experiment("Small CNN (from scratch)", model, tensors, n_classes, 32,
                             epochs, lr, batch, lambda m: m.train(), cb)
        st.session_state["cnn"] = {"res": res, "model": model, "sample": tensors[4][0]}
        remember(res)

    saved = st.session_state.get("cnn")
    if saved:
        res = saved["res"]
        st.success("Training finished (best epoch selected on the validation set).")
        show_metrics(res)
        a, b = st.columns(2)
        a.markdown("**Loss**")
        a.line_chart(res["history"][["train_loss", "val_loss"]])
        b.markdown("**Accuracy**")
        b.line_chart(res["history"][["train_acc", "val_acc"]])
        left, right = st.columns([1, 2])
        left.pyplot(plot_confusion(res["cm"], data["classes"], "Confusion matrix (test)"))
        with right:
            st.markdown("**Learned first-layer filters** (what the first convolution looks for)")
            st.pyplot(plot_filters(saved["model"]))
            st.markdown("**Feature maps** of one test image after conv1 + activation")
            st.pyplot(plot_feature_maps(saved["model"], saved["sample"]))


# --------------------------------------------------------------------------
# Tab 2: Transfer learning
# --------------------------------------------------------------------------
def build_pretrained(name, n_classes):
    set_seed()
    if name == "MobileNetV2":
        m = models.mobilenet_v2(weights="DEFAULT")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif name == "MobileNetV3-Small":
        m = models.mobilenet_v3_small(weights="DEFAULT")
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    else:
        m = models.resnet18(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def get_blocks(m, name):
    if name.startswith("MobileNet"):
        return list(m.features.children())
    return [m.layer1, m.layer2, m.layer3, m.layer4]


def head_module(m, name):
    if name.startswith("MobileNet"):
        return m.classifier
    return m.fc


def configure(m, name, strategy, k):
    for p in m.parameters():
        p.requires_grad = False
    for p in head_module(m, name).parameters():
        p.requires_grad = True
    if strategy == "finetune":
        for b in get_blocks(m, name)[-k:]:
            for p in b.parameters():
                p.requires_grad = True


def make_mode_fn(name, strategy, k):
    def fn(m):
        if strategy == "feature":
            m.eval()  # frozen backbone: BatchNorm statistics must stay fixed
        else:
            m.train()
            for b in get_blocks(m, name)[:-k]:
                b.eval()
            if name == "ResNet18":
                m.conv1.eval()
                m.bn1.eval()
    return fn


def transfer_tab(data):
    st.subheader("2-4. Transfer learning: feature extraction vs fine-tuning")
    st.markdown(
        "A network pre-trained on ImageNet (1.28M images) already knows edges, textures and "
        "shapes. **Feature extraction** freezes the whole backbone and trains only the new "
        "classifier head. **Fine-tuning** additionally unfreezes the last blocks (with a smaller "
        "learning rate) so the high-level features adapt to the new dataset."
    )
    c = st.columns(4)
    name = c[0].selectbox("Pre-trained model", TL_MODELS)
    size = c[1].selectbox("Input size", [64, 96, 128], index=0)
    epochs = c[2].slider("Epochs", 1, 10, 3, key="tl_ep")
    batch = c[3].selectbox("Batch size", [16, 32, 64], index=1, key="tl_bs")
    c = st.columns(3)
    lr_fe = c[0].select_slider("LR - feature extraction", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                               value=1e-3, format_func=lambda v: f"{v:g}", key="lr_fe")
    lr_ft = c[1].select_slider("LR - fine-tuning", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                               value=1e-4, format_func=lambda v: f"{v:g}", key="lr_ft")
    k = c[2].slider("Blocks to unfreeze (fine-tuning)", 1, MAX_UNFREEZE[name],
                    DEFAULT_UNFREEZE[name], key=f"unf_{name}")

    with st.expander("Validation protocol used for both strategies", expanded=False):
        st.markdown(
            "- One fixed **stratified 60/20/20 train/validation/test split** (seed 42)\n"
            "- Same input size, ImageNet normalisation, batch size, epochs, optimiser (Adam) and seed "
            "(identical head initialisation)\n"
            "- Only the **learning rate** differs (a head can use a large one; fine-tuning a small one "
            "to avoid destroying pre-trained weights)\n"
            "- The **best epoch is chosen on the validation set**; the **test set is touched only once** "
            "at the end\n"
            "- Reported: accuracy, macro precision/recall/F1, trainable parameters, training time, "
            "inference time"
        )

    if st.button("Run feature extraction + fine-tuning", type="primary"):
        n_classes = len(data["classes"])
        tensors = make_tensors(data, size, imagenet=True)
        results = {}
        try:
            for strategy, lr, label in (("feature", lr_fe, "Feature extraction"),
                                        ("finetune", lr_ft, "Fine-tuning")):
                st.write(f"**{label}** - {name}")
                with st.spinner("Loading pre-trained weights..."):
                    model = build_pretrained(name, n_classes)
                configure(model, name, strategy, k)
                cb = progress_ui()
                res = run_experiment(f"{name} - {label}", model, tensors, n_classes, size,
                                     epochs, lr, batch, make_mode_fn(name, strategy, k), cb)
                results[strategy] = res
                remember(res)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not finish: {exc}. (Pre-trained weights are downloaded on first use - "
                     "the server needs internet access.)")
            return
        st.session_state["tl"] = {"results": results, "name": name, "size": size, "k": k}

    saved = st.session_state.get("tl")
    if not saved:
        return
    fe, ft = saved["results"]["feature"], saved["results"]["finetune"]
    st.success(f"Comparison for {saved['name']} at {saved['size']}x{saved['size']} "
               f"(unfrozen blocks: {saved['k']}).")
    st.dataframe(pd.DataFrame([summary_row(fe), summary_row(ft)]).set_index("Model"))

    a, b = st.columns(2)
    val = pd.DataFrame({"Feature extraction": fe["history"]["val_acc"],
                        "Fine-tuning": ft["history"]["val_acc"]})
    trl = pd.DataFrame({"Feature extraction": fe["history"]["train_loss"],
                        "Fine-tuning": ft["history"]["train_loss"]})
    a.markdown("**Validation accuracy per epoch**")
    a.line_chart(val)
    b.markdown("**Training loss per epoch**")
    b.line_chart(trl)

    a, b = st.columns(2)
    a.pyplot(plot_confusion(fe["cm"], data["classes"], "Feature extraction (test)"))
    b.pyplot(plot_confusion(ft["cm"], data["classes"], "Fine-tuning (test)"))

    diff = ft["test_acc"] - fe["test_acc"]
    ratio = ft["train_time"] / max(fe["train_time"], 1e-6)
    better = "Fine-tuning" if diff > 0 else ("Feature extraction" if diff < 0 else "Both")
    st.markdown("#### Interpretation of this run")
    st.info(
        f"**{better}** gave the higher test accuracy "
        f"(fine-tuning minus feature extraction = {diff:+.4f}). Fine-tuning trained "
        f"{ft['trainable_params'] / 1e6:.2f}M parameters vs {fe['trainable_params'] / 1e6:.3f}M for "
        f"feature extraction and took {ratio:.1f}x as long. Fine-tuning usually wins when the new data "
        "differs from ImageNet (like these digits) and enough images are available; feature extraction "
        "is cheaper, safer on very small datasets and less prone to overfitting."
    )
    st.caption("Note: pre-trained models are used here as the transfer-learning backbone. "
               "VGG16 was left out on purpose: its weights are ~528 MB and ~15.5 GMACs/image, "
               "which does not fit free CPU hosting (see the Performance tab).")


# --------------------------------------------------------------------------
# Tab 3: Object detection
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading / loading detector weights...")
def load_detector(name):
    det = models.detection
    if name == DET_SSD:
        w = det.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        m = det.ssdlite320_mobilenet_v3_large(weights=w)
    else:
        w = det.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        m = det.fasterrcnn_mobilenet_v3_large_320_fpn(weights=w)
    m.eval()
    return m, w.meta["categories"]


@st.cache_data(show_spinner=False)
def download_sample():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    with urllib.request.urlopen(url, timeout=15) as r:
        return r.read()


def color_for(label_id):
    rng = np.random.default_rng(int(label_id) * 7919)
    return tuple(int(v) for v in rng.integers(60, 230, 3))


def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def draw_boxes(img, boxes, labels, scores, categories):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = get_font(max(12, img.width // 45))
    lw = max(2, img.width // 250)
    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        col = color_for(lab)
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        text = f"{categories[lab]} {sc:.2f}"
        tb = draw.textbbox((x1, y1), text, font=font)
        th = tb[3] - tb[1]
        top = max(0, y1 - th - 4)
        draw.rectangle([x1, top, x1 + (tb[2] - tb[0]) + 6, top + th + 4], fill=col)
        draw.text((x1 + 3, top + 1), text, fill="white", font=font)
    return out


def detection_tab():
    st.subheader("5. Object detection experiment (bounding boxes + confidence)")
    st.markdown(
        "Detectors predict **where** (bounding box) and **what** (COCO class) for every object, with a "
        "**confidence score**. Boxes below the confidence threshold are dropped. Pre-trained COCO "
        "models (80 classes) are used, so this is an inference experiment on any photo."
    )
    up = st.file_uploader("Upload a photo with objects (people, cars, animals...)",
                          type=[e.strip(".") for e in IMG_EXT], key="det_up")
    img = None
    if up is not None:
        try:
            img = Image.open(io.BytesIO(up.getvalue())).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read image: {exc}")
    if img is None:
        try:
            img = Image.open(io.BytesIO(download_sample())).convert("RGB")
            st.caption("Using a sample photo (downloaded). Upload your own to override.")
        except Exception:  # noqa: BLE001
            img = Image.fromarray(np.ascontiguousarray(load_sample_images().images[0]))
            st.caption("Sample download failed; using a built-in scikit-learn photo. "
                       "Upload an image with objects for a better demo.")
    img.thumbnail((800, 800))

    c1, c2 = st.columns([2, 1])
    chosen = c1.multiselect("Detectors to compare", [DET_SSD, DET_FRCNN], default=[DET_SSD])
    thr = c2.slider("Confidence threshold", 0.05, 0.95, 0.40, step=0.05)
    if not chosen:
        st.info("Select at least one detector.")
        return
    if not st.button("Run detection", type="primary"):
        st.image(img, caption="Input image")
        return

    cols = st.columns(len(chosen))
    for col, name in zip(cols, chosen):
        with col:
            st.markdown(f"**{name}**")
            try:
                model, cats = load_detector(name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load model: {exc}")
                continue
            tensor = torchvision.transforms.functional.to_tensor(img)
            with torch.no_grad():
                model([tensor])  # warm-up (first call is slower)
                t0 = time.time()
                out = model([tensor])[0]
                ms = (time.time() - t0) * 1000
            keep = out["scores"] >= thr
            boxes = out["boxes"][keep].numpy()
            labels = out["labels"][keep].numpy().astype(int)
            scores = out["scores"][keep].numpy()
            st.image(draw_boxes(img, boxes, labels, scores, cats))
            n_par = sum(p.numel() for p in model.parameters())
            m = st.columns(3)
            m[0].metric("Detections", len(boxes))
            m[1].metric("Inference", f"{ms:.0f} ms")
            m[2].metric("Params", f"{n_par / 1e6:.1f}M")
            if len(boxes):
                st.dataframe(pd.DataFrame({
                    "class": [cats[i] for i in labels], "confidence": scores.round(3),
                    "x1": boxes[:, 0].round(0), "y1": boxes[:, 1].round(0),
                    "x2": boxes[:, 2].round(0), "y2": boxes[:, 3].round(0),
                }))
            else:
                st.warning("No detection above the threshold - try lowering it.")
            st.session_state.setdefault("perf_det", {})[name] = {
                "Detector": name, "Params (M)": round(n_par / 1e6, 2),
                "Inference (ms, this image, CPU)": round(ms, 0), "Detections": len(boxes),
                "Mean confidence": round(float(scores.mean()), 3) if len(scores) else 0.0,
                "Threshold": thr,
            }
    st.markdown(
        "**Reading the result:** the box is the predicted object location, the label is the class and "
        "the number is the confidence (probability-like score). Lowering the threshold finds more "
        "objects but adds false positives; raising it gives fewer, more reliable boxes. "
        "SSDlite predicts boxes in a single pass (fast, lighter); Faster R-CNN first proposes regions, "
        "then classifies them (usually more accurate, slower)."
    )


# --------------------------------------------------------------------------
# Tab 4: Performance & computation
# --------------------------------------------------------------------------
REFERENCE = pd.DataFrame(
    [
        ["VGG16", "Classification", 138.4, 15.5, 528, "-"],
        ["ResNet50", "Classification", 25.6, 4.1, 98, "-"],
        ["ResNet18", "Classification", 11.7, 1.8, 45, "-"],
        ["MobileNetV2", "Classification", 3.5, 0.30, 14, "-"],
        ["MobileNetV3-Small", "Classification", 2.5, 0.06, 10, "-"],
        ["SSDlite320 MobileNetV3-L", "Detection", 3.4, 0.58, 13, 21.3],
        ["Faster R-CNN MobileNetV3-L 320 FPN", "Detection", 19.4, 0.72, 75, 22.8],
        ["Faster R-CNN ResNet50 FPN", "Detection", 41.8, 134.4, 160, 37.0],
    ],
    columns=["Model", "Task", "Params (M)", "GMACs (ImageNet 224 / COCO)", "Weights (MB)",
             "COCO box AP"],
)


def performance_tab():
    st.subheader("6. Performance and computational requirements")
    cls = st.session_state.get("perf_cls", {})
    det = st.session_state.get("perf_det", {})

    st.markdown("#### Measured in this session (CPU)")
    if cls:
        df = pd.DataFrame(list(cls.values())).set_index("Model")
        st.dataframe(df)
        best = df["Test acc"].idxmax()
        fastest = df["Inference (ms/img)"].idxmin()
        smallest = df["Total params (M)"].idxmin()
        cheapest = df["Train time (s)"].idxmin()
        st.markdown(
            f"- **Highest test accuracy:** {best} ({df.loc[best, 'Test acc']:.4f})\n"
            f"- **Fastest inference:** {fastest} ({df.loc[fastest, 'Inference (ms/img)']} ms/img)\n"
            f"- **Fewest parameters:** {smallest} ({df.loc[smallest, 'Total params (M)']} M)\n"
            f"- **Shortest training:** {cheapest} ({df.loc[cheapest, 'Train time (s)']} s)"
        )
    else:
        st.info("Train a model in the 'Small CNN' or 'Transfer Learning' tab to fill this table.")
    if det:
        st.markdown("**Detectors**")
        st.dataframe(pd.DataFrame(list(det.values())).set_index("Detector"))

    st.markdown("#### Reference: well-known models (approximate, from torchvision documentation)")
    st.dataframe(REFERENCE.set_index("Model"))
    st.caption("GMACs = billions of multiply-accumulates for one forward pass at the standard "
               "resolution (ImageNet 224x224 for classifiers, COCO ~800px / 320px for detectors).")

    st.markdown("#### Discussion")
    st.markdown(
        """
**Accuracy vs. cost.** A small CNN trained from scratch is tiny (tens of thousands of parameters,
milliseconds per image) and works well on simple data such as digits, but needs a lot of labelled data
on harder problems. Pre-trained backbones start with powerful features, so they reach high accuracy
with far fewer labelled images and epochs, at the price of more parameters and compute per image.

**Feature extraction vs. fine-tuning.** Feature extraction trains only the head: few trainable
parameters, no gradients through the frozen backbone, lowest memory and the fastest epochs. Fine-tuning
back-propagates through the unfrozen blocks, so each epoch costs more time and memory (roughly 2-3x),
and with small datasets it can overfit; it normally gives the best accuracy when the target domain
differs from ImageNet. A small learning rate and early selection on the validation set protect the
pre-trained weights.

**Computation scales with resolution.** Convolution cost grows with the number of pixels: doubling the
width and height multiplies the MACs by about 4. Using 64x64 inputs here makes CPU training practical,
but real photos usually need 224x224, where VGG16 (~15.5 GMACs) is about 50x more expensive than
MobileNetV2 (~0.3 GMACs) and has ~40x more parameters.

**Memory.** Parameters take 4 bytes each (fp32), so VGG16 needs ~528 MB just for weights, ResNet18
~45 MB and MobileNetV2 ~14 MB. Training also stores activations and gradients (and Adam keeps two extra
copies of every trainable parameter), so training needs several times the weight memory.
Mobile-friendly networks (depthwise-separable convolutions) are chosen for CPU and edge deployment.

**Hardware.** A GPU can be 10-100x faster than a CPU for training; on a CPU only (as on free Streamlit
hosting) use small models, small images, few epochs and a subset of the data. Quantisation (int8),
pruning and distillation reduce size and latency further for deployment.

**Detection.** Detectors are much more expensive than classifiers because they must localise many
objects: one-stage detectors (SSD/YOLO style) are fast, two-stage detectors (Faster R-CNN) are slower but
usually more accurate, and heavy backbones (ResNet50-FPN, ~134 GMACs) are impractical on CPU for
real-time use. Choose the model by the trade-off between required accuracy, latency and hardware.
"""
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.title("🧠 Deep Learning Vision Lab")
    st.caption("Small CNN • Transfer learning • Feature extraction vs fine-tuning • "
               "Object detection • Performance analysis")
    data = sidebar_data()

    t1, t2, t3, t4 = st.tabs(["Small CNN", "Transfer Learning", "Object Detection",
                              "Performance & Computation"])
    with t1:
        if data is None:
            st.info("Provide a valid dataset in the sidebar.")
        else:
            small_cnn_tab(data)
    with t2:
        if data is None:
            st.info("Provide a valid dataset in the sidebar.")
        else:
            transfer_tab(data)
    with t3:
        detection_tab()
    with t4:
        performance_tab()


if __name__ == "__main__":
    main()
