"""metaデータを加工するモジュール."""


def get_dtype(meta: dict) -> str:
    """dtypeを取得する."""
    if meta["PixelRepresentation"] == 0:
        if meta["BitsAllocated"] == 8:
            return "uint8"
        elif meta["BitsAllocated"] == 16:
            return "uint16"
        elif meta["BitsAllocated"] == 32:
            return "uint32"
        else:
            raise Exception("Unkonwn BitsAllocated.")
    elif meta["PixelRepresentation"] == 1:
        if meta["BitsAllocated"] == 8:
            return "int8"
        elif meta["BitsAllocated"] == 16:
            return "int16"
        elif meta["BitsAllocated"] == 32:
            return "int32"
        else:
            raise Exception("Unkonwn BitsAllocated.")
    elif meta["PixelRepresentation"] == 2:
        if meta["BitsAllocated"] == 32:
            return "float32"
        elif meta["BitsAllocated"] == 64:
            return "float64"
        else:
            raise Exception("Unkonwn BitsAllocated.")

    else:
        raise Exception("Unkonwn PixelRepresentation.")
