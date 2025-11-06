'''
Scopo:
    Verificare l'integrità delle immagini del dataset ModaNet e generare
    nuovi file di annotazioni COCO "puliti" solo con le immagini leggibili.
    Inoltre, eseguire uno split in training/validation (80/20 di default).

Cosa fa nello specifico:
    1) Carica il file COCO originale (modanet.json).
    2) Controlla per ogni immagine se il file esiste ed è leggibile con OpenCV.
       - Se l'immagine è assente o corrotta, viene esclusa.
    3) Mescola le immagini valide e crea due sottoinsiemi: train e val.
    4) Per ogni sottoinsieme, copia SOLO le annotazioni corrispondenti
       alle immagini rimaste e salva due JSON: modanet_train.json e modanet_val.json.
'''
import json
import random
import os
import cv2

### RUN ONLY ONCE per fixare le immagini broken nel dataset 

# Directory radice delle immagini e delle annotazioni
input_data = "./dataset/images"
input_ann = "./dataset/annotations"
# Nome del file COCO originale che contiene tutte le immagini/annotazioni
ann_file = "modanet.json"
# Proporzioni di split train/val
train_ratio = 0.8
val_ratio = 0.2

# Carica il JSON COCO di partenza
with open(os.path.join(input_ann, ann_file)) as f:
    data = json.load(f)
# Estrae i tre campi principali previsti dallo standard COCO
images = data['images']
annotations = data['annotations']
categories = data['categories']

# Filtra le immagini non valide: mancanti su disco o corrotte/non leggibili
valid_images = []
broken_images_count = 0
for img in images:
    img_path = os.path.join(input_data, img['file_name'])
    if os.path.exists(img_path):
        try:
            image_data = cv2.imread(img_path)
            if image_data is not None:
                valid_images.append(img)
            else:
                broken_images_count += 1
                print(f"File corrotto o non leggibile: {img_path}")
        except Exception as e:
            # Qualsiasi eccezione in lettura conta come immagine non valida
            broken_images_count += 1
            print(f"Errore durante la lettura del file {img_path}: {e}")
    else:
        # File immagine non presente su disco
        broken_images_count += 1
# Report sintetico sull'operazione di pulizia
print(f"Original images count: {len(images)}")
print(f"Broken images found: {broken_images_count}")
print(f"Valid images count: {len(valid_images)}")
# Da questo punto in avanti si lavora solo con immagini valide
images = valid_images
categories = data['categories']
# Mescola l'ordine delle immagini per uno split casuale
random.shuffle(images)

n = len(images)
n_train = int(n * train_ratio)
n_val = int(n * val_ratio)
# Suddivisione in liste: prime n_train per il training, le successive n_val per la validation
train_imgs = images[:n_train]
val_imgs = images[n_train:n_train + n_val]


def subset(images_subset):
    """
    Crea un sotto-dataset COCO mantenendo:
      - images: solo quelle nello subset,
      - annotations: solo quelle con image_id presente nel subset,
      - categories: copia delle categorie originali,
      - campi 'info' e 'licenses' minimi per compatibilità.
    """
    img_ids = {img["id"] for img in images_subset}
    anns = [a for a in annotations if a["image_id"] in img_ids]
    return {"images": images_subset, "annotations": anns, "categories": categories,
            "info":[{'description': 'ModaNet Dataset'}], "licenses": []}

# Costruisce i due split (train/val)
splits = {
    "train": subset(train_imgs),
    "val":   subset(val_imgs),
}

# Salva i nuovi file di annotazioni COCO nella directory di annotazioni
# I file generati sono: modanet_train.json e modanet_val.json
for name, data in splits.items():
    with open(f"{input_ann}/modanet_{name}.json", "w") as f:
        json.dump(data, f)
    print(f"{name}: {len(data['images'])} images, {len(data['annotations'])} annotations")




'''# ==============================================================
# APPENDICE: analisi e visualizzazione dopo il "fix_images"
# (attivare se si vuole verificare visivamente l'effetto del cleaning)
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt

try:
    # Conteggi generali
    orig_count = len(images)
    valid_count = len(valid_images)
    broken_count = broken_images_count

    # Riepilogo per split
    train_count = len(train_imgs)
    val_count = len(val_imgs)

    print("\n--- REPORT POST FIX_IMAGES ---")
    print(f"Immagini originali: {orig_count}")
    print(f"Immagini valide:    {valid_count}")
    print(f"Immagini rotte:     {broken_count}")
    print(f"Train split:        {train_count}")
    print(f"Val split:          {val_count}")

    # Costruzione DataFrame per il riepilogo
    df_fix = pd.DataFrame({
        "Tipo": ["Totali originali", "Valide dopo fix", "Corrotte/escluse", "Train split", "Validation split"],
        "Numero": [orig_count, valid_count, broken_count, train_count, val_count]
    })

    # --- Grafico a barre ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_fix["Tipo"], df_fix["Numero"], color=["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"])
    plt.title("Riepilogo del cleaning delle immagini (fix_images)")
    plt.ylabel("Numero di immagini")
    plt.xticks(rotation=20, ha="right")

    # Etichette sopra le barre
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height, f"{int(height)}", 
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("fix_images_summary.png", dpi=300)
    plt.show()

    # Salvataggio CSV opzionale
    df_fix.to_csv("fix_images_summary.csv", index=False)
    print("\n[Riepilogo salvato in fix_images_summary.png e fix_images_summary.csv]")

except Exception as e:
    print(f"[ERRORE] Non è stato possibile generare il riepilogo: {e}")
'''