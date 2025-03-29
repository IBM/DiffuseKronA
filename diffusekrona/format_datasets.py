import os, shutil

path_to_dataset = "../data"

path_to_delete = [os.path.join(path_to_dataset, i, 'generated') for i in os.listdir(path_to_dataset)]
for path in path_to_delete:
    if os.path.exists(path): shutil.rmtree(path)