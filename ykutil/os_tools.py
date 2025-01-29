import os


def search_file_in_folder(fname: str, folder: str):
    if os.path.isfile(fname):
        return fname
    assert os.path.isdir(folder), f"{folder} is not a folder"

    findings = []
    for root, _, files in os.walk(folder):
        if fname in files:
            findings.append(os.path.join(root, fname))

    assert len(findings) > 0, f"{fname} not found in {folder}"
    assert len(findings) == 1, f"{fname} found in multiple places: {findings}"
    return findings[0]
