import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ZIP_NAME = os.path.join(ROOT, 'ADS_DWH_Project_trimmed.zip')

EXCLUDE_DIRS = {
    'data', 'venv', '.git', '__pycache__', '.venv', '.pytest_cache', '.mypy_cache', 'node_modules'
}
EXCLUDE_FILES = {'.env'}
EXCLUDE_SUFFIXES = {'.pyc', '.pyo', '.log'}

print(f"Creating zip at: {ZIP_NAME}")
with zipfile.ZipFile(ZIP_NAME, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # compute relative dir from root
        rel_dir = os.path.relpath(dirpath, ROOT)
        if rel_dir == '.':
            rel_dir = ''
        # skip excluded directories
        parts = rel_dir.split(os.sep) if rel_dir else []
        if any(p in EXCLUDE_DIRS for p in parts):
            # modify dirnames in-place to prevent walking excluded subdirs
            dirnames[:] = []
            continue

        # filter out excluded dirnames so os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for fname in filenames:
            if fname in EXCLUDE_FILES:
                continue
            if any(fname.endswith(suf) for suf in EXCLUDE_SUFFIXES):
                continue

            abs_path = os.path.join(dirpath, fname)
            # skip the output zip if it exists in the tree
            if os.path.abspath(abs_path) == os.path.abspath(ZIP_NAME):
                continue

            arcname = os.path.join(rel_dir, fname) if rel_dir else fname
            zf.write(abs_path, arcname)

print('Zip creation complete:')
print(ZIP_NAME)
size = os.path.getsize(ZIP_NAME)
print(f'Size: {size / (1024*1024):.2f} MB')
