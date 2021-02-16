def label_transform(dsets_dict):
    idx_to_classId = dsets_dict.idx_to_class
    className_to_classId = dsets_dict.className_to_classId
    classId_to_className = {v:int(k) for k,v in className_to_classId.items()}
    idx_to_className = {k:classId_to_className[v] for k,v in idx_to_classId.items()}
    return idx_to_className