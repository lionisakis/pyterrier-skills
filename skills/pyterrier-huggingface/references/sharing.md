# Sharing and storing artifacts

Checked against the PyTerrier 1.1.2 source. See also the
[Artifact how-to](https://pyterrier.readthedocs.io/en/latest/artifacts/how-to.html).

## Contents
1. How an artifact is stored
2. Hugging Face
3. Zenodo
4. Peer-to-peer
5. Manual hosting

## 1. How an artifact is stored

An artifact is a folder on disk (an index, a cache, ...) with a `pt_meta.json` file that describes it:

```json
{"type": "sparse_index", "format": "terrier", "package_hint": "python-terrier"}
```

- `type` and `format` pick the class that loads it, through the `pyterrier.artifact` entry point named
  `<type>.<format>`. `package_hint` is the pip package to suggest if that class is missing. Some
  classes add extra fields, such as the model checkpoint or vector size.
- `pt_meta.json` is written by the artifact class itself. Older folders without it are recognised by
  metadata adapters, e.g. a Terrier index that has a `data.properties` file.
- For distribution, the folder is packed into `artifact.tar.lz4`: a tar archive with files in sorted
  order and no timestamps, compressed with LZ4, so the same index gives the same file.
- `artifact.tar.lz4.json` sits next to the archive. It records the total size, the contents and the
  SHA-256 hash, which is checked on download. Large archives are split into `artifact.tar.lz4.0`,
  `.1`, …, and the `.json` file lists the parts.

## 2. Hugging Face

```python
index.to_hf("owner/name.terrier")                  # main branch
index.to_hf("owner/name.terrier@v2")               # a branch, or branch="v2"
index.to_hf("owner/name.terrier", private=True)    # private repository
```

- Needs `huggingface_hub` and a token with write access (`HF_TOKEN`, or `huggingface-cli login`).
- Creates a **dataset** repository if it doesn't exist, and uploads the archive in parts of up to 20 GB.
- Writes a README card with the tags `pyterrier`, `pyterrier-artifact`, `pyterrier-artifact.<type>` and
  `pyterrier-artifact.<type>.<format>`. That's how the tag search finds it. Edit the card afterwards:
  describe the corpus, the settings, the model for dense indexes, and how to load it.
- Name repositories `<collection>.<kind>`, following the existing ones: `fiqa.terrier`, `fiqa.pisa`,
  `fiqa.retromae.flex`, `fiqa.splade-v3.cache`.
- Publishing makes the index public unless `private=True`. Confirm the name, the visibility, and that
  the collection's licence allows it, before uploading.

## 3. Zenodo

```python
index.to_zenodo(pretty_name="FiQA Terrier index")   # needs ZENODO_TOKEN
index.to_zenodo(sandbox=True)                        # test against sandbox.zenodo.org
pt.Artifact.from_zenodo("<record id>")               # or pt.Artifact.from_url("zenodo:<record id>")
```

`to_zenodo` uploads the archive and leaves a **draft**. Publish it yourself through the link it prints.
A published Zenodo record gets a DOI and can't be deleted.

## 4. Peer-to-peer

```python
index.to_p2p()                                       # prints a Magic Wormhole code
pt.Artifact.from_p2p("<code>", "./received.terrier")  # on the other machine
```

Needs `pip install magic-wormhole` on both machines. The target path must not exist yet.

## 5. Manual hosting

```python
path = index.build_package("my-index.tar.lz4")   # also writes my-index.tar.lz4.json
pt.Artifact.from_url("https://example.org/my-index.tar.lz4")
```

Host both files side by side. `from_url` reads the `.json` file when it is present, to check the hash
and find split parts.
