from enum import Enum


class Sensors(Enum):
    S1 = "S1"
    S2 = "S2"
    s2 = "s2"


class S1(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []


class S2(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B8A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]  # no guarda name, guarda value !
    NONE = []


class s2(Enum):
    B1 = aerosol = 1
    B2 = blue = 2
    B3 = green = 3
    B4 = red = 4
    B5 = re1 = 5
    B6 = re2 = 6
    B7 = re3 = 7
    B8 = nir1 = 8
    B8A = nir2 = 9
    B9 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
    RGB = [B4, B3, B2]  # no guarda name, guarda value !
    NONE = []
