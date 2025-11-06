'''import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# === Parametri da personalizzare ===
ANNOTATIONS_PATH = Path("./dataset/annotations/modanet.json")
OUTPUT_PNG = Path("modanet_categories.png")   # output per presentazioni (300 dpi)
OUTPUT_SVG = Path("modanet_categories.svg")   # output vettoriale per slide

# === Caricamento e conteggio categorie ===
with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Costruisci mapping id->nome categoria
# ModaNet usa tipicamente ID consecutivi, ma è più robusto costruire un dict
id_to_name = {}
for c in data.get("categories", []):
    # "id" e "name" sono le chiavi standard
    id_to_name[c["id"]] = c["name"]

# Conta occorrenze per category_id
counts = Counter()
for ann in data.get("annotations", []):
    cid = ann["category_id"]
    counts[cid] += 1

# Trasforma in DataFrame ordinato per frequenza
rows = [(id_to_name.get(cid, str(cid)), n) for cid, n in counts.items()]
df = pd.DataFrame(rows, columns=["Categoria", "Occorrenze"]).sort_values("Occorrenze", ascending=False)

# === Plot: bar chart a colori per presentazioni ===
plt.figure(figsize=(12, 6))

# Palette: uso tab20 estendendola se servono più colori
base_colors = plt.get_cmap("tab20").colors
colors = (base_colors * (len(df) // len(base_colors) + 1))[:len(df)]

bars = plt.bar(df["Categoria"], df["Occorrenze"], color=colors)

# Etichette sopra le barre
for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height, f"{int(height)}",
             ha="center", va="bottom", fontsize=10)

# Titolo e assi leggibili in slide
plt.title("ModaNet: frequenza delle categorie", fontsize=16, pad=12)
plt.ylabel("Numero di occorrenze", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()

# Salvataggi: PNG ad alta risoluzione e SVG vettoriale
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT_SVG, bbox_inches="tight")
plt.show()

# Conta quante immagini uniche ci sono nel dataset
num_images = len({ann["image_id"] for ann in data["annotations"]})
print("Numero di immagini nel dataset fornito:", num_images)

print("Numero totale di annotazioni:", len(data["annotations"]))'''

'''import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image
from pathlib import Path

# === Parametri ===
ANNOTATIONS_PATH = Path("./dataset/annotations/modanet.json")
IMAGES_DIR = Path("./dataset/images")
OUTPUT_PATH = Path("annotated_example.png")

# === Caricamento JSON ===
with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Mappa ID categoria → nome
id_to_name = {c["id"]: c["name"] for c in data["categories"]}

# Raggruppa annotazioni per immagine
annotations_by_image = {}
for ann in data["annotations"]:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# === Scegli un'immagine casuale con più oggetti (meglio per slide) ===
candidate_images = [img_id for img_id, anns in annotations_by_image.items() if len(anns) >= 3]
image_id = random.choice(candidate_images)

# Trova il nome file corrispondente
image_info = next(img for img in data["images"] if img["id"] == image_id)
image_path = IMAGES_DIR / image_info["file_name"]

# === Carica immagine ===
image = Image.open(image_path).convert("RGB")

# === Plot ===
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.axis("off")

# Colori distinti per ogni categoria
unique_cats = set(ann["category_id"] for ann in annotations_by_image[image_id])
color_map = {cid: plt.cm.tab20(i % 20) for i, cid in enumerate(unique_cats)}

# Disegna le annotazioni
for ann in annotations_by_image[image_id]:
    cid = ann["category_id"]
    color = color_map[cid]

    # Disegna poligono se disponibile
    if "segmentation" in ann and len(ann["segmentation"]) > 0:
        for seg in ann["segmentation"]:
            poly = Polygon(
                [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)],
                closed=True,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.35
            )
            ax.add_patch(poly)

    # Disegna bounding box
    if "bbox" in ann:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

    # Etichetta categoria
    cat_name = id_to_name[cid]
    ax.text(x, y - 3, cat_name, color=color, fontsize=10, weight="bold", backgroundcolor="white")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Immagine annotata salvata in: {OUTPUT_PATH}")'''


'''import pandas as pd
import matplotlib.pyplot as plt

# Inserisci i valori corretti (puoi leggerli anche da un file o copiarli qui)
orig_count = 52377          # immagini totali originali
valid_count = 46868         # immagini valide dopo il fix (esempio)
broken_count = orig_count - valid_count
train_count = int(valid_count * 0.8)
val_count = int(valid_count * 0.2)

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

for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height, f"{int(height)}",
             ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("fix_images_summary.png", dpi=300)
plt.show()

df_fix.to_csv("fix_images_summary.csv", index=False)
print("\n[Riepilogo salvato in fix_images_summary.png e fix_images_summary.csv]")
'''



'''# save as make_training_table_pretty.py
import matplotlib.pyplot as plt
import pandas as pd

# Dati
data = {
    "Parametro": [
        "Optimizer",
        "Scheduler",
        "Learning rate iniziale",
        "Batch size",
        "Epoche",
    ],
    "Valore": ["SGD", "StepLR", "0.005", "4", "20"],
    "Note": [
        "Con momentum = 0.9",
        "Decadimento del learning rate",
        "Testato anche 0.001–0.01",
        "Limitato da memoria GPU",
        "Training su HPC",
    ],
}

df = pd.DataFrame(data)

# Stile grafico
fig, ax = plt.subplots(figsize=(12, 1.6))
ax.axis("off")

# Colori e parametri estetici
header_color = "#2E3A87"
cell_color_1 = "#EEF1FA"
cell_color_2 = "#FFFFFF"
text_color_header = "white"
text_color_body = "#222222"
border_color = "#C5CAE9"

# Creazione tabella
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    loc="center",
)

# Stile celle
for (row, col), cell in table.get_celld().items():
    # Header
    if row == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color=text_color_header, weight="bold", size=14)
    else:
        cell.set_facecolor(cell_color_1 if row % 2 == 0 else cell_color_2)
        cell.set_text_props(color=text_color_body, size=13)
    cell.set_edgecolor(border_color)
    cell.set_linewidth(1)

# Regolazioni proporzioni
table.auto_set_font_size(False)
table.scale(1.5, 2.0)

# Salvataggio in PNG
plt.savefig(
    "training_params_pretty.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2,
    facecolor="white",
)
plt.close()

print(" Immagine salvata come 'training_params_pretty.png'")'''


import matplotlib.pyplot as plt
import pandas as pd

# Dati del best model
data = {
    "LR": [0.005],
    "Epoche": [20],
    "H_box": [128],
    "H_mask": [128],
    "Fixed": ["True"],
    "AP_box": [0.361],
    "AP_mask": [0.318],
}

# Crea il DataFrame
df = pd.DataFrame(data)

# Crea la figura
fig, ax = plt.subplots(figsize=(10, 1.5), dpi=300)
ax.axis("off")

# Colori
header_color = "#2E3A87"
body_color = "#F4F6FB"
border_color = "#C5CAE9"

# Crea la tabella
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    loc="center",
)

# Stile celle
for (r, c), cell in table.get_celld().items():
    if r == 0:  # Header
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold", size=12)
    else:  # Riga dati
        cell.set_facecolor(body_color)
        cell.set_text_props(color="black", size=11)
    cell.set_edgecolor(border_color)
    cell.set_linewidth(1)

# Impostazioni estetiche
table.auto_set_font_size(False)
table.scale(1.3, 2.2)  # Aumenta leggibilità

# Titolo sopra la tabella
plt.text(
    0.5,
    1.25,
    "Modello migliore (Best Model)",
    ha="center",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    color="#2E3A87",
    transform=ax.transAxes,
)

# Salva la tabella
plt.savefig(
    "best_model_table.png",
    bbox_inches="tight",
    pad_inches=0.3,
    facecolor="white",
)
plt.close()

print("✅ Immagine salvata come 'best_model_table.png'")
