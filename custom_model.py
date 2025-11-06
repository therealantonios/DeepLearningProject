'''
    Definisce teste personalizzate per Mask R-CNN (sia per le bbox che per le maschere)
    e fornire una funzione di factory per costruire un modello Mask R-CNN basato su
    ResNet50+FPN con queste teste custom.
'''

from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn


class CustomBoxPredictor(nn.Module):
    def __init__(self, in_features, units, num_classes):
        super(CustomBoxPredictor, self).__init__()

        # Testa di classificazione personalizzata:
        # - Trasforma il vettore di caratteristiche 'in_features' in 'units' neuroni,
        #   applica ReLU + Dropout per regolarizzare,
        #   quindi proietta su 'num_classes' (incluso lo sfondo).
        self.cls_score = nn.Sequential(
            nn.Linear(in_features, units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(units, num_classes)  # L'ultimo layer deve avere num_classes in output
        )

        # Testa di regressione bbox personalizzata:
        # - Stessa idea: MLP poco profondo che produce, per OGNI classe,
        #   4 parametri di box (x1, y1, x2, y2) → output = num_classes * 4
        self.bbox_pred = nn.Sequential(
            nn.Linear(in_features, units),
            nn.ReLU(),
            nn.Linear(units, num_classes * 4)  # 4 output per ogni classe: [x1, y1, x2, y2]
        )

    def forward(self, x):
        # L'head della ROI in torchvision di solito passa un tensore 2D (N, in_features).
        # Se per qualche motivo arriva 4D (N, C, H, W), lo appiattiamo a partite dalla dimensione 1
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        scores = self.cls_score(x)      # logits di classe: (N, num_classes)
        bbox_deltas = self.bbox_pred(x) # regressioni bbox: (N, num_classes*4)
        return scores, bbox_deltas


class CustomMaskPredictor(nn.Module):
    def __init__(self, in_channels, hidden_layer, num_classes):
        super(CustomMaskPredictor, self).__init__()

        # Testa maschere personalizzata:
        # - Quattro conv 3x3 con padding=1 per preservare la risoluzione (stride=1),
        #   tutte con ReLU. Profondità interna = hidden_layer.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_layer, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_layer, hidden_layer, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_layer, hidden_layer, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_layer, hidden_layer, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Ultimo strato per upsampling e predizione delle maschere:
        # - ConvTranspose2d con kernel=2 e stride=2 raddoppia H e W (upsample x2),
        #   producendo una mappa per classe: output shape (N, num_classes, H*2, W*2).
        self.last_conv = nn.ConvTranspose2d(hidden_layer, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        # Passaggio attraverso i layer convoluzionali e upsampling finale
        x = self.conv_layers(x)
        x = self.last_conv(x)
        return x


def build_custom_maskrcnn(num_classes: int, box_units: int = 128, mask_hidden: int = 32):
    # Crea un Mask R-CNN base con backbone ResNet50+FPN.
    # 'weights="COCO_V1"' carica pesi pre-addestrati su COCO (utile per trasferimento).
    model = maskrcnn_resnet50_fpn(weights="COCO_V1")

    # Custom box head
    # Recuperiamo 'in_features' dalla testa originale e istanziamo la nostra variante MLP.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = CustomBoxPredictor(in_features, box_units, num_classes)

    # Custom mask head
    # Recuperiamo i canali in input della conv finale originale per mantenere la compatibilità.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = CustomMaskPredictor(in_features_mask, mask_hidden, num_classes)

    return model
