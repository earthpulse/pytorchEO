import numpy as np
from tqdm import tqdm

LABELS = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sport and leisure facilities",
    "Non-irrigated arable land",
    "Permanently irrigated land",
    "Rice fields",
    "Vineyards",
    "Fruit trees and berry plantations",
    "Olive groves",
    "Pastures",
    "Annual crops associated with permanent crops",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Bare rock",
    "Sparsely vegetated areas",
    "Burnt areas",
    "Inland marshes",
    "Peatbogs",
    "Salt marshes",
    "Salines",
    "Intertidal flats",
    "Water courses",
    "Water bodies",
    "Coastal lagoons",
    "Estuaries",
    "Sea and ocean",
]

LABELS7 = {
    "Urban": [0, 1, 2],
    "Crop": [11, 12, 13, 14, 15, 16, 18, 17, 19, 20],
    "Forest": [21, 22, 23, 24],
    "Vegetation": [25, 31, 26, 27, 28],
    "Beaches, dunes, sands": [29],
    "Wetlands": [33, 34, 35, 36],
    "Water": [38, 39, 40, 41, 42],
}

LABELS19 = {
    "Urban fabric": [0, 1],
    "Industrial or commercial units": [2],
    "Arable land": [11, 12, 13],
    "Permanent crops": [14, 15, 16, 18],
    "Pastures": [17],
    "Complex cultivation patterns": [19],
    "Land principally occupied by agriculture, with significant areas of natural vegetation": [
        20
    ],
    "Agro-forestry areas": [21],
    "Broad-leaved forest": [22],
    "Coniferous forest": [23],
    "Mixed forest": [24],
    "Natural grassland and sparsely vegetated areas": [25, 31],
    "Moors, heathland and sclerophyllous vegetation": [26, 27],
    "Transitional woodland, shrub": [28],
    "Beaches, dunes, sands": [29],
    "Inland wetlands": [33, 34],
    "Coastal wetlands": [35, 36],
    "Inland waters": [38, 39],
    "Marine waters": [40, 41, 42],
}


def oh2labels(label_groups, oh):
    if label_groups:
        labels = list(label_groups.keys())
        return [labels[ix] for ix in np.nonzero(oh.cpu().numpy())[0]]
    return [LABELS[ix] for ix in np.nonzero(oh.cpu().numpy())[0]]


def group_labels(labels, groups):
    new_labels = []
    for multi_label in tqdm(labels):
        new_multi_label = []
        for label in multi_label:
            original_id = LABELS.index(label)
            for group in groups:
                if original_id in groups[group] and group not in new_multi_label:
                    new_multi_label.append(group)
                    break
        new_labels.append(new_multi_label)
    return new_labels


def encode_labels(labels, unique_labels):
    new_labels = []
    for multi_label in tqdm(labels):
        new_multi_label = np.zeros(len(unique_labels), dtype=int)
        ixs = [unique_labels.index(label) for label in multi_label]
        new_multi_label[ixs] = 1
        new_labels.append(new_multi_label.tolist())
    return new_labels


def get_processed_data_filename(
    processed_data_name, label_groups, filter_clouds=False, filter_snow=False
):
    file_name = processed_data_name
    if filter_clouds:
        file_name += "_cloudless"
    if filter_snow:
        file_name += "_snowless"
    if label_groups:
        file_name += f"_LABELS{len(label_groups.keys())}"
    file_name += ".json"
    return file_name
