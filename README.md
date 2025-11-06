# 🧠 Object Detection and Instance Segmentation on ModaNet Dataset

Progetto sviluppato per il corso **Deep Learning and Generative Models – University of Parma**  
Autore: **Antonio Signorelli**  
A.A. 2024/2025  

---

## 📌 Descrizione del progetto

L’obiettivo del progetto è stato lo **sviluppo completo di una pipeline di Object Detection e Instance Segmentation** basata su **Mask R-CNN** applicata al dataset **ModaNet**, un dataset fashion in formato COCO-like.

Il lavoro è stato svolto interamente da zero, includendo:
- la preparazione e correzione del dataset,
- la definizione di un **Custom Dataset Loader** e di un **modello Mask R-CNN personalizzato**,
- l’addestramento su infrastruttura **HPC dell’Ateneo**,
- e la valutazione delle performance tramite metriche COCO standard (AP).

---

## 🧩 Struttura del progetto

```bash
├── custom_loader.py         # Definizione dataset COCO-like e data augmentation
├── custom_model.py          # Architettura Mask R-CNN (ResNet50 + FPN) con teste custom
├── train.py                 # Script di training con checkpoint e scheduler
├── evaluation.py            # Valutazione automatica con COCOeval
├── plot_results.py          # Generazione grafici comparativi (AP_box / AP_mask)
├── prediction.py            # Inferenza e visualizzazione predizioni su immagini nuove
├── fix_images.py            # Pulizia immagini corrotte e creazione split train/val
├── fix_annotations.py       # Correzione automatica annotazioni errate (footwear/boots)
├── check_annotations.py     # Confronto annotazioni vecchie vs corrette
└── output/                  # Risultati, CSV metriche e grafici


