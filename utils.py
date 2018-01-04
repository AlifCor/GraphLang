import os, codecs

def load_text(path, encoding = "utf8"):
    paths = path.split("/")
    with codecs.open(os.path.join(paths[0], *paths[1:]),encoding=encoding) as f:
        text = f.read()
    return text