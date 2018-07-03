from pymatgen import MPRester

if __name__ == "__main__":
    MAPI_KEY = None  #Materials API key!
    QUERY = "mp-1203"  
    # QUERY = "TiO"  
    # QUERY = "Ti-O" 

    mpr = MPRester(MAPI_KEY) 

    structures = mpr.get_structures(QUERY)
    for s in structures:
        print s
