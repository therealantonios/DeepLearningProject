'''
Scopo:
    Caricare un modello Mask R-CNN con teste custom dai checkpoint del progetto
    ed eseguire predizioni (bbox + maschere) su un insieme di immagini, salvando
    i risultati sovrapposti alle immagini originali.

Funzionalità principali:
    - get_best_checkpoint: legge "output/results.csv" e seleziona il checkpoint
      con AP_box più alto.
    - predict: esegue l'inferenza su una singola immagine, disegnando bbox,
      maschere e punteggi direttamente sull'immagine.
    - main: carica il modello in base al checkpoint, scansiona una cartella di
      immagini e salva le predizioni in output.
'''

import argparse
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from custom_model import build_custom_maskrcnn

# --- CONFIGURAZIONE ---

# Nomi delle classi ModaNet
CLASS_NAMES = [
    'bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses',
    'pants', 'top', 'shorts', 'skirt', 'headwear', 'scarf/tie'
]
# Numero di classi totali (13 + 1 per lo sfondo)
NUM_CLASSES = 14

# Mappa classe->colore per disegno: colori vividi e leggibili su sfondo fotografico
COLORS = {name: (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255)) for name in CLASS_NAMES}


def get_best_checkpoint(results_csv_path="output/results.csv"):
    """
    Trova il nome del checkpoint con il miglior AP_box dal file dei risultati.
    """
    try:
        import pandas as pd
        df = pd.read_csv(results_csv_path, sep=';')
        best_model_row = df.loc[df['AP_box'].idxmax()]
        print(
            f"Trovato miglior modello in base a AP_box: {best_model_row['name']} (AP_box: {best_model_row['AP_box']:.4f})")
        return best_model_row['name']
    except (FileNotFoundError, ImportError, KeyError):
        print(
            "Attenzione: Impossibile trovare il miglior checkpoint da 'results.csv'. Specificalo manualmente con --checkpoint.")
        return None


def predict(model, image_path, device, score_threshold):
    """
    Esegue la predizione su una singola immagine e disegna bbox/maschere/punteggi.
    """
    # Carica l'immagine e la converte in RGB
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Errore: Impossibile leggere l'immagine {image_path}")
        return None, None
    # Converti in RGB per la trasformazione torchvision
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Trasformazione base: ToTensor normalizza a [0,1] e cambia canali in [C,H,W]
    transform = T.ToTensor()
    image_tensor = transform(img_rgb).to(device)

    # Esegui l'inferenza
    model.eval()
    with torch.no_grad():
        # Il modello si aspetta un batch (lista) di immagini
        prediction = model([image_tensor])

    # Estrai i risultati (scatole, etichette, punteggi, maschere)
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    masks = (prediction[0]['masks'].cpu().numpy() > 0.5).squeeze(1)

    # Disegna i risultati sull'immagine originale (BGR)
    for i in range(len(boxes)):
        if scores[i] > score_threshold:
            # Estrai le informazioni
            box = boxes[i]
            label_id = labels[i]
            score = scores[i]
            mask = masks[i]

            # Ottieni il nome della classe (sottrai 1 perché le etichette partono da 1)
            class_name = CLASS_NAMES[label_id - 1] if (label_id - 1) < len(CLASS_NAMES) else f"ID:{label_id}"
            color = COLORS.get(class_name, (0, 255, 0))

            # Disegna il bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Disegna la maschera con una certa trasparenza tramite addWeighted
            overlay = img.copy()
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, -1)   # -1: riempie i contorni
            alpha = 0.4
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Etichetta + score vicino all'angolo in alto a sinistra della bbox
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img


def main():
    """
    Punto di ingresso dello script:
      - individua il checkpoint migliore (o quello passato a mano),
      - costruisce il modello e carica i pesi,
      - scorre le immagini in input e salva le predizioni in output.
    """
    parser = argparse.ArgumentParser(description="Prediction on a set of images")
    # Percorsi di default: cartella con nuove immagini e cartella di salvataggio risultati
    input_dir = "./dataset/new_images"
    output_dir = "./output/predictions"
    # Soglia score predefinita: 0.5 è un buon compromesso per rumore vs recall
    threshold = 0.5

    # Determina il dispositivo (GPU o CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Dispositivo in uso: {device}")
    
    # Se non specificato via CLI, prova a ricavare il miglior checkpoint dal CSV
    checkpoint_name = None
    if checkpoint_name is None:
        checkpoint_name = get_best_checkpoint()
        if checkpoint_name is None:
            return

    checkpoint_path = Path("./runs") / checkpoint_name
    # Verifica esistenza del file di checkpoint
    if not checkpoint_path.exists():
        print(f"Errore: File del checkpoint non trovato in '{checkpoint_path}'")
        return

    # Estrai gli iperparametri dal nome del file per costruire il modello
    match = re.search(r"hbox(\d+)_h(\d+)", checkpoint_name)
    if not match:
        print("Errore: Impossibile estrarre gli iperparametri dal nome del checkpoint.")
        print("Il nome deve contenere 'hbox<num>_h<num>'.")
        return

    hbox_units = int(match.group(1))
    h_units = int(match.group(2))

    # Costruisci modello e carica pesi dal checkpoint
    print(f"Caricamento del modello da '{checkpoint_path}'...")
    model = build_custom_maskrcnn(num_classes=NUM_CLASSES, box_units=hbox_units, mask_hidden=h_units)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print("Modello caricato con successo.")

    # Crea la cartella di output se non esiste
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Trova tutte le immagini nella cartella di input
    image_files = list(Path(input_dir).glob('*.[jp][pn]g'))  # Cerca .jpg, .jpeg, .png
    if not image_files:
        print(f"Nessuna immagine trovata in '{input_dir}'. Controlla il percorso e le estensioni.")
        return

    # Esegui le predizioni su ogni immagine
    for image_path in tqdm(image_files, desc="Predizione sulle immagini"):
        predicted_image = predict(model, image_path, device, threshold)

        if predicted_image is not None:
            # Salva l'immagine con le predizioni
            output_path = Path(output_dir) / image_path.name
            cv2.imwrite(str(output_path), predicted_image)

    print(f"\nPredizioni completate. Le immagini sono state salvate in '{output_dir}'.")


if __name__ == '__main__':
    main()