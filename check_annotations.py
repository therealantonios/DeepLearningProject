'''
Script per confrontare due file di annotazioni in formato COCO. Per ogni categoria,
vengono contate le annotazioni comuni e quelle uniche a ciascun file.
'''
import json

# CONFRONTA IL DATASET "FIXED" CON QUELLO ORIGINALE PER VERIFICARE LE DIFFERENZE

file1 = "./dataset/annotations/modanet_train.json"
file2 = "./dataset/annotations/fixed_modanet_train.json"

# Caricamento dei file JSON
with open(file1) as f:
    coco1 = json.load(f)

with open(file2) as f:
    coco2 = json.load(f)

# Estrazione di annotazioni e categorie da ciascun file (formato COCO)
annotations1 = coco1["annotations"]
annotations2 = coco2["annotations"]
categories1 = coco1["categories"]
categories2 = coco2["categories"]

print(categories1)

# Costruzione di una mappa nome-id per le categorie di ciascun file.
cats1 = {c["name"]: c["id"] for c in categories1}
cats2 = {c["name"]: c["id"] for c in categories2}
results = []

# Itera su tutte le classi del primo file 
for classe in sorted(cats1.keys()):
    id1 = cats1[classe]
    id2 = cats2.get(classe)

    print(f"\nClass: '{classe}'")

    if id2 is None:
        print(f"  Class '{classe}' not found in {file2}")
        continue

    # Filtra le annotazioni di entrambe le sorgenti per la classe corrente
    # Si selezionano le annotazioni con category_id == id della classe corrispondente
    ann1 = [a for a in annotations1 if a.get("category_id") == id1]
    ann2 = [a for a in annotations2 if a.get("category_id") == id2]

    print(f"  Annotations: {len(ann1)} in {file1}, {len(ann2)} in {file2}")

    # Confronto basato su (image_id, bbox)
    # Per confrontare la "stessa" annotazione nei due file si usa la coppia (image_id, bbox).
    # bbox è convertita in tuple per poter essere hashabile ed entrare in un set.
    set1 = {(a.get("image_id"), tuple(a.get("bbox", []))) for a in ann1}
    set2 = {(a.get("image_id"), tuple(a.get("bbox", []))) for a in ann2}

    solo_in_1 = set1 - set2 # annotazioni presenti nel file1 e non nel file2
    solo_in_2 = set2 - set1 # annotazioni presenti nel file2 e non nel file1
    comuni = set1 & set2 # annotazioni comuni ad entrambi i file

    # Report sintetico per la classe corrente
    print(f"  Common: {len(comuni)}")
    print(f"  Only in {file1}: {len(solo_in_1)}")
    print(f"  Only in {file2}: {len(solo_in_2)}")
    #results.append({"Classe": classe, "File1": len(set1), "File2": len(set2)})

'''# ==============================================================
# APPENDICE: visualizzazione grafica delle differenze tra dataset
# (attivare se si vogliono plottare i risultati del confronto)
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt

# Raccogli i risultati del loop precedente (deve essere eseguito prima)
# Se il loop originale stampa solo i valori, puoi aggiungere una lista `results` e fare append()
# Esempio (nel tuo loop): results.append({"Classe": classe, "File1": len(set1), "File2": len(set2)})

# Esegui questa parte solo se hai creato la lista 'results'
try:
    df = pd.DataFrame(results)
except NameError:
    print("\n[AVVISO] Nessuna variabile 'results' trovata. Per attivare il grafico, aggiungi nel loop principale:")
    print("         results = []  (prima del for)  e  results.append({...})  dentro al for.")
else:
    # Calcola differenze
    df["Differenza"] = df["File2"] - df["File1"]
    df = df.sort_values("Differenza", ascending=False)

    # --- Grafico 1: differenze (File2 - File1)
    plt.figure(figsize=(10, max(6, 0.4 * len(df))))
    colors = ["#2ca02c" if d > 0 else "#d62728" if d < 0 else "#7f7f7f" for d in df["Differenza"]]
    plt.barh(df["Classe"], df["Differenza"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Differenza di annotazioni (file2 - file1)")
    plt.title("Differenze tra dataset Fixed e Originale per categoria")
    plt.tight_layout()
    plt.savefig("diff_dataset_fixed_vs_original.png", dpi=300)
    plt.show()

    # --- Grafico 2: confronto assoluto
    plt.figure(figsize=(10, max(6, 0.4 * len(df))))
    width = 0.4
    x = range(len(df))
    plt.barh([i + width/2 for i in x], df["File1"], height=width, label="Originale")
    plt.barh([i - width/2 for i in x], df["File2"], height=width, label="Fixed")
    plt.yticks(list(x), df["Classe"])
    plt.xlabel("Numero di annotazioni")
    plt.title("Confronto assoluto di annotazioni per categoria")
    plt.legend()
    plt.tight_layout()
    plt.savefig("abs_dataset_fixed_vs_original.png", dpi=300)
    plt.show()'''


