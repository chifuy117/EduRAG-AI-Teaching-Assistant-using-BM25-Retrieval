import zipfile
zf='ADS_DWH_Project_trimmed.zip'
with zipfile.ZipFile(zf) as z:
    names=z.namelist()
    for n in names:
        print(n)
    print('---')
    print('Total files:', len(names))
