'''
Scopo:
    Script di training per Mask R-CNN con teste personalizzate sul dataset ModaNet.
    Gestisce:
      - parsing degli iperparametri da riga di comando (lr, epochs, batch, ecc.),
      - costruzione dei DataLoader (train/val) con trasformazioni di base,
      - creazione del modello con teste custom e ottimizzatore + scheduler,
      - validazione per epoca tramite `evaluate_loss`,
      - salvataggio e ripresa tramite checkpoint.
'''
import random
import time
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

import argparse
from torch import optim
import torch
import os
from tqdm import tqdm

from custom_loader import ModanetDataset, SimpleTransforms, collate_fn, save_checkpoint, evaluate
from custom_model import build_custom_maskrcnn

# ---- ARGPARSE: parsing iperparametri da CLI ----
parser = argparse.ArgumentParser(description="Train Mask R-CNN on Modanet Dataset")
parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=20, help='Total number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--hidden-layer-box', type=int, default=8, help='Hidden layer size for the box predictor')
parser.add_argument('--hidden-layer-mask', type=int, default=8, help='Hidden layer size for the mask predictor')
parser.add_argument('--fixed', type=bool, default=False, help='Use fixed annotations')
args = parser.parse_args()

# Bind degli argomenti a variabili locali per leggibilità
hidden_layer_box = args.hidden_layer_box
hidden_layer_mask = args.hidden_layer_mask
learning_rate = args.lr
epochs = args.epochs
fixed = args.fixed
batch_size = args.batch_size # massimo 8 su HPC per evitare crash OOM

# Numero di worker per DataLoader (0: evita problemi di multiprocessing su alcuni ambienti/HPC)
NUM_WORKERS = 0

# ---- PATH E COSTANTI ----
input_data = "./dataset/images"
input_ann = "./dataset/annotations"
output_dir = "./output"

# Split train/val
train_ratio = 0.8
val_ratio = 0.2

# 13 classes + 1 background
num_classes = 14

# Seeds per la riproducibilità
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ---- CHECKPOINT E SCELTA DEI FILE DI ANNOTAZIONE ----
# Se `--fixed` è attivo, usa le annotazioni corrette e marca il checkpoint con suffisso "_fixed"
if fixed:
    checkpoint_name = f"last_hbox{hidden_layer_box}_h{hidden_layer_mask}_lr{learning_rate}_e{epochs}_fixed.ckpt"
    ann_train = os.path.join(input_ann, "fixed_modanet_train.json")
    ann_val = os.path.join(input_ann, "fixed_modanet_val.json")
else:
    checkpoint_name = f"last_hbox{hidden_layer_box}_h{hidden_layer_mask}_lr{learning_rate}_e{epochs}.ckpt"
    ann_train = os.path.join(input_ann, "modanet_train.json")
    ann_val = os.path.join(input_ann, "modanet_val.json")

# ---- DEVICE ----
# Usa GPU se disponibile; altrimenti CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ---- DATASET E DATALOADER ----
train_ds = ModanetDataset(input_data, ann_train, SimpleTransforms())
val_ds = ModanetDataset(input_data, ann_val, SimpleTransforms(train=False))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS,
                          pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS,
                        pin_memory=True)

# ---- MODELLO E OTTIMIZZAZIONE ----
# Costruisce Mask R-CNN con teste custom (box/mask) e numero classi configurato
model = build_custom_maskrcnn(num_classes=num_classes, box_units=hidden_layer_box, mask_hidden=hidden_layer_mask)
# Parametri trainabili 
params = [p for p in model.parameters() if p.requires_grad]
# Ottimizzatore
optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
# Scheduler: StepLR diminuisce il LR di un fattore 'gamma' (default 0.1) ogni 'step_size' epoche
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4)
# Loss scaler
scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
# Sposta il modello sul device scelto
model.to(device)

# ---- TRAINING PHASE & CHECKPOINTING ----
checkpoint_path = Path("./runs") / checkpoint_name
best_map = -1.0
start_epoch = 1

# Caricamento checkpoint se presente (resume)
if checkpoint_path.is_file():
    print(f"=> Caricamento del checkpoint '{checkpoint_path}'")
    # Carica il checkpoint. 'map_location' assicura che funzioni
    # anche se lo carichi su un device diverso (es. da GPU a CPU)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(checkpoint['best_metric'])
    # Ripristina lo stato del modello
    model.load_state_dict(checkpoint['model'])

    # Ripristina lo stato dell'ottimizzatore
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Imposta l'epoca di partenza a quella successiva all'ultima salvata
    start_epoch = checkpoint['epoch'] + 1

    # Ripristina il miglior punteggio ottenuto finora
    best_map = checkpoint['best_metric']

    print(f"=> Checkpoint caricato. Si riparte dall'epoca {start_epoch} con best_map: {best_map:.4f}")
else:
    print(f"=> Nessun checkpoint trovato in '{checkpoint_path}', si inizia da zero.")

# ---- LOOP DI TRAINING PER EPOCHE ----
for epoch in range(start_epoch, epochs + 1):
    model.train()

    loss_hist = []
    start = time.time()
    # plot progress
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{20}")
    for i, (images, targets) in enumerate(pbar):
        # Sposta batch su device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Autocast (float16/float32) solo se scaler è attivo (CUDA)
        with torch.amp.autocast(enabled=(scaler is not None), device_type='cuda'):
            # Per Mask R-CNN, il forward in training restituisce un dict di loss componenti
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            # Backward/step in mixed precision
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        loss_hist.append(losses.item())

        dur = time.time() - start # Durata dall'inizio dell'epoca (aggiorna a ogni iterazione)

    # ---- METRICHE DI EPOCA ----
    tr_loss, tr_time = np.mean(loss_hist) if loss_hist else 0.0, dur
    # Validazione: `evaluate` deve iterare sul val_loader e calcolare una loss media
    val_loss = evaluate(model, val_loader, device)
    # Step dello scheduler (dopo ogni epoca)
    lr_scheduler.step()

    print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} ({tr_time / 60:.1f} min), val_loss={val_loss:.4f}")
    # Salvataggio checkpoint dell'epoca corrente
    save_checkpoint(Path("./runs") / checkpoint_name, model, optimizer, epoch, best_map)
