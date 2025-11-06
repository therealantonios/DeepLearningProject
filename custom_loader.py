'''
Custom DataLoader e funzioni di utilità per il training e valutazione su dataset in formato COCO.
Include:
- ModanetDataset: Dataset personalizzato per caricare immagini e annotazioni COCO.
- SimpleTransforms: Trasformazioni semplici per data augmentation (flip orizzontale).
- collate_fn: Funzione di collate per DataLoader.
- evaluate: Funzione per valutare il modello su un dataset COCO.
- visualize_predictions: Funzione per salvare immagini con predizioni e ground truth sovrapposte.
'''
from pathlib import Path
from typing import Any, Dict
import numpy as np, cv2, torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pycocotools.coco import COCO
import random
import os
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from pycocotools import mask as mask_utils


class ModanetDataset(Dataset):
    def __init__(
            self,
            images_dir,
            ann_file,
            transforms,
            filter_min_size=1,
            cache=True,
            class_whitelist=None,
            subset_size=None,
    ):
        # Percorso della directory che contiene le immagini
        self.images_dir = Path(images_dir)

        # parser COCO con il file annotazioni (formato COCO)
        self.coco = COCO(str(ann_file))
        # Lista di tutti gli ID immagine presenti nelle annotazioni COCO
        self.img_ids = list(self.coco.imgs.keys())
        # Trasformazioni da applicare alle immagini e ai target
        self.transforms = transforms
        # Filtro per rimuovere bbox troppo piccole (lato minimo in pixel)
        self.filter_min_size = filter_min_size
        # Flag per eventuale caching immagini 
        self.cache = cache
        # Eventuale whitelist di categorie (category_id COCO) da includere
        self.class_whitelist = set(class_whitelist) if class_whitelist is not None else None
        # Cache immagini in RAM per immagini caricate 
        self._img_cache: Dict[int, np.ndarray] = {}

        # Se si vuole un sottoinsieme (debug/rapidi test), usa i primi N id per riproducibilità
        if subset_size is not None:
            self.img_ids = self.img_ids[:subset_size]
            print(f"--- ATTENZIONE: Dataset ridotto a {len(self.img_ids)} immagini per test. ---")

        # Costruisci mapping continuo delle categorie: COCO usa id sparsi, molti modelli vogliono etichette 1..K
        cats = sorted(self.coco.cats.keys())
        if self.class_whitelist is not None:
            # Se whitelist presente, tieni solo i category_id inclusi
            cats = [c for c in cats if c in self.class_whitelist]
        # Mappa da id COCO a etichetta contigua [1..K]
        self.cat_id_to_contig = {cid: i + 1 for i, cid in enumerate(cats)}
        # Mappa inversa: da etichetta contigua a category_id COCO
        self.contig_to_cat_id = {v: k for k, v in self.cat_id_to_contig.items()}

    def __len__(self):
        return len(self.img_ids) # Dimensione del dataset = numero di immagini considerate

    def _load_image(self, img_id):
        # Caricatore di immagini robusto a piccole incongruenze tra JSON e file system
        # Recupera le info dell'immagine (incluso file_name) a partire dall'id
        info = self.coco.loadImgs([img_id])[0]
        fname = str(info.get("file_name", "")).strip()
        fname = Path(fname).name

        # Genera una lista di percorsi candidati da provare in sequenza
        candidates = []
        # 1) Join diretto directory immagini + nome file
        candidates.append(self.images_dir / fname)
        # 2) Stesso nome base con estensioni comuni (case-insensitive) per tollerare mismatch di estensione/maiuscole
        stem = Path(fname).stem
        common_exts = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
        for ext in common_exts:
            p = self.images_dir / f"{stem}{ext}"
            candidates.append(p)
        # 3) Se il JSON conteneva un path relativo interno a images_dir, prova anche quello
        if Path(info.get("file_name", "")).parent != Path('.'):
            candidates.append(self.images_dir / Path(info["file_name"]))

        img = None
        # Scansiona i candidati e carica la prima immagine che esiste e si legge correttamente
        for p in candidates:
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    break
        if img is None:
            # Se non si trova nulla, segnala errore con elenco puntuale dei tentativi fatti
            tried = [str(p) for p in candidates]
            raise FileNotFoundError(
                f"Image not found for id={img_id}. Tried: \n" + "\n".join(tried)
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _ann_to_target(self, img_id, h, w):
        """
        Converte le annotazioni COCO associate a img_id in un `target` nello
        standard torchvision:
            - boxes: Tensor [N,4] in formato [x1, y1, x2, y2]
            - labels: Tensor [N] con etichette contigue [1..K]
            - masks: Tensor [N,H,W] con maschere binarie
            - image_id, iscrowd, area
        """
        # Ricava gli id delle annotazioni associati a questa immagine
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        # Carica gli oggetti annotazione completi per questi id
        anns = self.coco.loadAnns(ann_ids)

        # Liste di accumulo per bbox, etichette, maschere, flag crowd e aree
        boxes, labels, masks, iscrowd, areas = [], [], [], [], []

        for a in anns:
            # Filtra per whitelist
            if self.class_whitelist is not None and a["category_id"] not in self.class_whitelist:
                continue

            # Se la categoria non è nel mapping (può capitare con whitelist), skippa
            if a["category_id"] not in self.cat_id_to_contig:
                continue

            x, y, bw, bh = a["bbox"]
            if bw < self.filter_min_size or bh < self.filter_min_size:
                continue

            # Converti bbox nel formato [x1, y1, x2, y2] atteso dai modelli di torchvision
            boxes.append([x, y, x + bw, y + bh])
            # Mappa il category_id COCO nella label contigua che userà il modello
            labels.append(self.cat_id_to_contig[a["category_id"]])
            # iscrowd: campo COCO per indicare annotazioni "crowd"; default 0
            iscrowd.append(a.get("iscrowd", 0))
            # area: se non specificata, calcola come w*h (approssimazione accettata)
            areas.append(a.get("area", bw * bh))

            # Gestione della segmentazione (può essere None, lista di poligoni o RLE)
            seg = a.get("segmentation", None)
            if seg is None:
                # Se non c'è segmentazione, crea una maschera vuota
                masks.append(np.zeros((h, w), dtype=np.uint8))
            elif isinstance(seg, list):
                # Segmentazione poligonale: riempi la maschera unendo i poligoni
                m = np.zeros((h, w), dtype=np.uint8)
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(m, [pts], 1)
                masks.append(m)
            elif isinstance(seg, dict) and ("counts" in seg):
                # Segmentazione RLE (Run Length Encoding)
                # Se manca "size", costruisci un oggetto RLE a partire dal dict
                rle = seg if seg.get("size") else mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(rle)
                # Se m ha più componenti (H, W, N), riduci con OR
                if m.ndim == 3:
                    m = m.any(axis=2).astype(np.uint8)
                masks.append(m)
            else:
                # Formato inatteso: fallback a maschera vuota per non interrompere la pipeline
                masks.append(np.zeros((h, w), dtype=np.uint8))

        # Conversione delle liste in tensori torch con i dtypes/shape attesi dai modelli
        if len(boxes) == 0:
            # Caso senza annotazioni valide: crea tensori vuoti con shape coerenti
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            masks_t = torch.tensor(np.stack(masks), dtype=torch.uint8)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
        
        # Dizionario target nel formato convenzionale torchvision (Detection/Segmentation)
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id]),
            "iscrowd": iscrowd_t,
            "area": areas_t,
        }
        return target

    def __getitem__(self, idx):
        # Restituisce (img, target) per l'indice richiesto.
        # Recupera l'id immagine dal vettore di id e carica l'immagine
        img_id = self.img_ids[idx]
        img = self._load_image(img_id)
        # Dimensioni H, W per costruzione maschere coerenti
        h, w = img.shape[:2]
        # Costruisci il target a partire dalle annotazioni COCO per questa immagine
        target = self._ann_to_target(img_id, h, w)

        if self.transforms:
            img, target = self.transforms(img, target)

        img = T.ToTensor()(img)
        return img, target


# In fase di training, può applicare con probabilità 0.5 un flip orizzontale per data augmentation
class SimpleTransforms:
    def __init__(self, train=True, hflip_p=0.5):
        self.train = train      # True se in modalità training (abilita augmentation)
        self.hflip_p = hflip_p  # Probabilità di flip orizzontale

    def __call__(self, img: np.ndarray, target: Dict[str, Any]):
        if self.train and random.random() < self.hflip_p:
            # Flip orizzontale di tutta l'immagine e aggiornamento delle bbox e maschere
            img = np.ascontiguousarray(np.fliplr(img))
            w = img.shape[1]
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
            # Se sono presenti maschere, flippale orizzontalmente (asse W)
            if target["masks"].numel() > 0:
                masks = target["masks"].numpy()[:, :, ::-1].copy()
                target["masks"] = torch.tensor(masks, dtype=torch.uint8)
        return img, target


def collate_fn(batch):
    # evita errori quando il numero di oggetti cambia tra immagini.
    # Funzione di collate per DataLoader: converte lista di tuple in tuple di liste
    # Risultato: ( [img1, img2, ...], [target1, target2, ...] )
    return tuple(zip(*batch))


def evaluate(model, data_loader, device):  # Evaluate_loss vecchia che poi è diventata solo evaluate (riga 168 di train.py)
    # Valuta il modello su un data_loader stile COCO, restituendo AP per bbox e per maschere
    model.eval()
    coco_gt = data_loader.dataset.coco # Ground truth COCO ricavato dal dataset
    results = [] # Lista di detezioni in formato COCO per pycocotools

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Sposta tutte le immagini del batch su device
            images = [img.to(device) for img in images]
            # Inference: outputs è una lista di dict (uno per immagine)
            outputs = model(images)
            # Qui si assume batch size 1 (si prende la prima/sola uscita)
            output = outputs[0]
            # Recupera l'id dell'immagine dal target corrispondente
            image_id = int(targets[0]["image_id"])  
            # # Estrai tensori e convertili in numpy per composizione del risultato
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"].cpu().numpy()  # (N,1,H,W)

            for i in range(len(boxes)):
                bx = boxes[i].tolist()
                x1, y1, x2, y2 = bx
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                seg = mask_utils.encode(np.asfortranarray((masks[i, 0] > 0.5).astype(np.uint8)))
                seg["counts"] = seg["counts"].decode("utf-8")  # pycocotools expects str
                results.append({
                    "image_id": image_id,
                    "category_id": int(data_loader.dataset.contig_to_cat_id[int(labels[i])]),
                    "bbox": bbox,
                    "score": float(scores[i]),
                    "segmentation": seg,
                })

    if len(results) == 0:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(results)

    # Boxes AP

    coco_eval_box = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval_box.evaluate()
    coco_eval_box.accumulate()
    coco_eval_box.summarize()
    ap_box = float(coco_eval_box.stats[0])

    # Masks AP
    coco_eval_mask = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval_mask.evaluate()
    coco_eval_mask.accumulate()
    coco_eval_mask.summarize()
    ap_mask = float(coco_eval_mask.stats[0])

    return ap_box, ap_mask


def save_checkpoint(path, model, optimizer, epoch,
                    best_metric=None):
    
    """
    Salva uno snapshot del training contenente:
        - pesi del modello,
        - stato dell'optimizer,
        - epoca corrente,
        - metrica migliore
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))


def visualize_predictions(model, data_loader, device, output_dir, num_images=10, score_threshold=0.5, class_names=None):
    '''    
    Esegue inferenza su alcune immagini del data_loader e salva, per ognuna, un'immagine affiancata:
      - a sinistra: predizioni (bbox, contorni maschere, score/label),
      - a destra : ground truth (bbox/label).
    '''
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Genera colori casuali per ogni classe
    if class_names:
        colors = {name: (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255)) for name in
                  class_names}
    else:
        colors = {}

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= num_images:
                break

            image_tensor = images[0].to(device)
            target = targets[0]

            # Ottieni predizioni
            outputs = model([image_tensor])
            output = outputs[0]

            # Converte l'immagine da tensore a formato OpenCV (H, W, C) e da RGB a BGR
            img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Crea due copie: una per le predizioni, una per il ground truth
            img_pred = img_bgr.copy()
            img_gt = img_bgr.copy()

            # --- Disegna le PREDZIONI sull'immagine ---
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = (output["masks"].cpu().numpy() > 0.5).squeeze(1)  # (N, H, W)

            for j, box in enumerate(boxes):
                if scores[j] > score_threshold:
                    class_id = labels[j]
                    class_name = class_names[class_id - 1] if class_names and (class_id - 1) < len(
                        class_names) else f"Class {class_id}"
                    color = colors.get(class_name, (0, 255, 0))

                    # Disegna Bounding Box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)

                    # Disegna Maschera (come contorno)
                    contours, _ = cv2.findContours(masks[j].astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_pred, contours, -1, color, 2)

                    # Scrivi Etichetta e Score
                    label_text = f"{class_name}: {scores[j]:.2f}"
                    cv2.putText(img_pred, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Disegna il GROUND TRUTH sull'altra immagine ---
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()

            for j, box in enumerate(gt_boxes):
                class_id = gt_labels[j]
                class_name = class_names[class_id - 1] if class_names and (class_id - 1) < len(
                    class_names) else f"Class {class_id}"
                color = colors.get(class_name, (255, 0, 0))  # Colore diverso per GT
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_gt, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Unisci le due immagini (predizione a sinistra, ground truth a destra)
            combined_image = np.hstack((img_pred, img_gt))

            # Aggiungi titoli
            cv2.putText(combined_image, "Predictions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(combined_image, "Ground Truth", (img_pred.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Salva l'immagine
            image_id = int(target["image_id"])
            output_filename = os.path.join(output_dir, f"img_{image_id}_pred.jpg")
            cv2.imwrite(output_filename, combined_image)

    print(f"Immagini di visualizzazione salvate in: {output_dir}")
