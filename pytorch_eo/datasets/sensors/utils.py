def bands2values(bands):
    if isinstance(bands, list):
        if len(bands) == 1:
            return bands[0].value
        else:
            return [band.value for band in bands]
    else:
        return bands.value


# convert bands from enum to names


def bands2names(bands):
    if isinstance(bands, list):
        if len(bands) == 1:
            if isinstance(bands[0].value, list):
                return [band.name for band in bands[0].value]
            else:
                return [bands[0].name]
        else:
            return [band.name for band in bands]
    else:
        if isinstance(bands.value, list):
            return [band.name for band in bands.value]
        else:
            return bands.name
