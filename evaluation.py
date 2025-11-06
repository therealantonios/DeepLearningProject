'''
Scopo:
    Caricare automaticamente i checkpoint dei modelli addestrati e eseguire:
      - visualizzazione qualitativa delle predizioni (salvataggio immagini),
      - valutazione quantitativa (AP box e AP mask tramite COCOeval),
    e infine salvare un riepilogo dei risultati in CSV.

Flusso generale:
    1) Scansione della cartella CHECKPOINT_DIR per i file ".ckpt".
    2) Parsing degli iperparametri dal nome file tramite regex (hbox, h, lr, epoche, fixed).
    3) Ricostruzione del modello con le teste custom e caricamento degli stati.
    4) Preparazione del dataset di validazione (versione "fixed" o originale).
    5) Visualizzazione di alcune predizioni (salvate in VISUALIZATION_DIR).
    6) Valutazione AP box e AP mask con pycocotools (funzione 'evaluate').
    7) Salvataggio risultati in "output/results.csv" e stampa ordinata per AP box.
'''
import argparse
import os
import random
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from custom_loader import collate_fn, ModanetDataset, SimpleTransforms, evaluate, visualize_predictions
from custom_model import build_custom_maskrcnn

# Numero di worker per i DataLoader; 0 per evitare problemi di multiprocessing su alcune macchine/HPC
NUM_WORKERS = 0

# Selezione automatica del device: preferisci CUDA se disponibile
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Directory di output per immagini di visualizzazione e per checkpoint
VISUALIZATION_DIR = "./output/visualizations"
CHECKPOINT_DIR = "./runs"
# Nomi delle 13 categorie ModaNet
CLASS_NAMES = [
    'bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses',
    'pants', 'top', 'shorts', 'skirt', 'headwear', 'scarf/tie'
]

# Path base di immagini e annotazioni
input_data = "./dataset/images"
input_ann = "./dataset/annotations"
ann_val = os.path.join(input_ann, "modanet_val.json")
ann_val_fixed = os.path.join(input_ann, "fixed_modanet_val.json")

seed = 42 # Semi per riproducibilità (random, NumPy, PyTorch)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_classes = 14 # Numero di classi nel modello (13 classi + eventuale sfondo secondo convenzione torchvision)

batch_size = 2 # Batch size: empiricamente limitato (max 8 su HPC) per evitare out-of-memory/crash

# Regex per estrarre iperparametri dal nome del checkpoint.
pattern = re.compile(
    r"last_hbox(\d+)_h(\d+)_lr([0-9.]+)_e(\d+)(_fixed)?"
)

models = {}
print("Loading models...", flush=True)
# Scansione directory dei checkpoint
for file_name in os.listdir(CHECKPOINT_DIR):
    if file_name.endswith(".ckpt"):
        match = pattern.match(file_name.replace(".ckpt", ""))
        if not match:
            continue

        # Parsing degli iperparametri da gruppi catturati
        hbox_units = int(match.group(1))            # unità MLP della testa bbox
        h_units = int(match.group(2))               # canali hidden della testa mask
        lr = float(match.group(3))                  # learning rate usato in training
        epochs = int(match.group(4))                # numero di epoche
        is_fixed = bool(match.group(5))             # True se il nome contiene "_fixed"

        checkpoint_path = os.path.join(CHECKPOINT_DIR, file_name)
        print(f"Model path: {checkpoint_path}")
        print(f"hbox={hbox_units}, h={h_units}, lr={lr}, epochs={epochs}, fixed={is_fixed}")

        # Ricostruzione del modello con le teste custom e stesso num_classes
        model = build_custom_maskrcnn(num_classes=num_classes, box_units=hbox_units, mask_hidden=h_units)
        # Caricamento stato dal checkpoint (pesato su device corrente)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model'])

        # Memorizza il modello e i relativi metadati
        models[file_name] = {
            "model": model,
            "params": {
                "hbox": hbox_units,
                "h": h_units,
                "lr": lr,
                "epochs": epochs,
                "fixed": is_fixed
            }
        }

print("Evaluating models...")
results_summary = [] # lista di dizionari per assemblare il DataFrame finale

# Loop su tutti i modelli caricati
for name, entry in models.items():
    print(f"\nEvaluating {name} ...")
    model = entry["model"].to(device)
    params = entry["params"]

    # Directory in cui salvare le immagini di visualizzazione per questo modello
    model_viz_dir = os.path.join(VISUALIZATION_DIR, name.replace('.ckpt', ''))

    # Se il checkpoint è marcato come "fixed", usa il validation set corretto; altrimenti quello originale
    # subset_size=20 per velocizzare visualizzazione/valutazione (campionamento fisso dei primi ID)
    if params['fixed']:
        val_ds = ModanetDataset(input_data, ann_val_fixed, SimpleTransforms(train=False), subset_size=20)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        val_ds = ModanetDataset(input_data, ann_val, SimpleTransforms(train=False), subset_size=20)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # DataLoader separato (batch size implicito 1) per generare immagini qualitative
    val_loader_viz = DataLoader(val_ds, shuffle=False, collate_fn=collate_fn)

    # Salva alcune immagini con predizioni (affiancate a GT) per ispezione qualitativa
    visualize_predictions(model=model, data_loader=val_loader_viz, device=device, output_dir=model_viz_dir,
                          num_images=5, class_names=CLASS_NAMES)
    
    # DataLoader per la valutazione quantitativa (AP box/mask)
    val_loader_eval = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    ap_box, ap_mask = evaluate(model, val_loader_eval, device)

    # Accumula risultati per il riepilogo
    results_summary.append({
        "name": name,
        "hbox": params["hbox"],
        "h": params["h"],
        "lr": params["lr"],
        "epochs": params["epochs"],
        "fixed": params["fixed"],
        "AP_box": ap_box,
        "AP_mask": ap_mask
    }) 
    torch.cuda.empty_cache()

# Costruzione del DataFrame dei risultati e salvataggio su CSV (separator ';' per compatibilità locale)
df = pd.DataFrame(results_summary)
output_csv_path = "output/results.csv"
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df.to_csv(output_csv_path, sep=';', index=False)

# Stampa un riepilogo ordinato per AP_box decrescente
print("\n--- Results Summary ---")
print(df.sort_values(by="AP_box", ascending=False))