from enum import Enum


class Sensors(Enum):
    S1 = "S1"
    S2 = "S2"


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
