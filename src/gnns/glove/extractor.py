import zipfile
with zipfile.ZipFile("./glove840b300dtxt.zip", 'r') as zip_ref:
    zip_ref.extractall(".")