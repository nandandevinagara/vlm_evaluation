import zipfile
import sys

zip_filepath = sys.argv[1]

with zipfile.ZipFile(zip_filepath, "r") as z:
    names = z.namelist()
    for each_name in names[:10]:
        print(each_name)  # Preview
    z.extract(names[5], path="extracted/")
