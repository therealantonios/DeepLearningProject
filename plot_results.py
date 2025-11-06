'''
Scopo:
    Leggere il file dei risultati (output/results.csv) prodotto dallo script di
    valutazione e generare un grafico comparativo tra le metriche di performance
    (AP_box e AP_mask) dei diversi modelli
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Data path
RESULTS_FILE = "output/results.csv"
OUTPUT_IMAGE_FILE = "output/results_comparison.png"


def plot_model_comparison(results_path: str, output_path: str):
    """
    Legge i risultati da un file CSV e crea un grafico a barre per confrontare
    le performance (AP_box e AP_mask) dei modelli.
    """
    # 1. Carica i dati dal file CSV
    try:
        df = pd.read_csv(results_path, sep=';')
    except FileNotFoundError:
        print(f"Errore: Il file '{results_path}' non è stato trovato.")
        print("Assicurati di aver eseguito lo script di validazione per generarlo.")
        return

    # 2. Prepara i dati per il plotting
    # Ordina i risultati per AP_box decrescente per una visualizzazione più chiara
    df = df.sort_values(by="AP_box", ascending=False).reset_index(drop=True)

    # Crea etichette più leggibili per l'asse X del grafico
    df['label'] = df.apply(
        lambda row: f"HBox:{row['hbox']}, HMsk:{row['hmask']}\nLR:{row['lr']}, Epochs:{row['epochs']}, Fix:{row['fixed']}",
        axis=1
    )

    labels = df['label']
    ap_box = df['AP_box']
    ap_mask = df['AP_mask']

    x = np.arange(len(labels))  # Posizioni delle etichette sull'asse X
    width = 0.35  # Larghezza delle barre

    # 3. Crea il grafico
    # Aumentiamo le dimensioni per una migliore leggibilità
    fig, ax = plt.subplots(figsize=(18, 9))

    # Crea le barre per AP_box e AP_mask
    rects1 = ax.bar(x - width / 2, ap_box, width, label='AP Box', color='skyblue')
    rects2 = ax.bar(x + width / 2, ap_mask, width, label='AP Mask', color='salmon')

    # 4. Aggiungi etichette, titolo e legenda
    ax.set_ylabel('Average Precision (AP)')
    ax.set_title('Performance comparison for AP Box and AP mask', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend()

    # Aggiungi i valori numerici sopra ogni barra
    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=8)

    # Imposta i limiti dell'asse Y per dare più spazio alle etichette
    ax.set_ylim(0, max(df['AP_box'].max(), df['AP_mask'].max()) * 1.2)

    # Ottimizza il layout per evitare sovrapposizioni
    fig.tight_layout()

    # 5. Salva il grafico su file
    # Assicurati che la cartella di output esista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)

    print(f"Grafico salvato con successo in: '{output_path}'")
    # plt.show() # Decommenta questa linea se vuoi visualizzare il grafico subito


if __name__ == '__main__':
    plot_model_comparison(RESULTS_FILE, OUTPUT_IMAGE_FILE)