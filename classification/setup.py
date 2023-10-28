import os
import zipfile

# link https://drive.google.com/drive/folders/1xgnCJ5WJVEhROZRCFKuu3mDmn4rCi2-C?usp=drive_link

os.mkdir('data')
os.mkdir('logs')
os.mkdir('logs/fit')
os.mkdir('model')
os.mkdir('data_temp')

data = os.path.join('drive')

for i in os.listdir(data):
    file = os.path.join(data,i)
    folder_name = i.split('.')[0]
    path = os.path.join('data_temp',folder_name)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(path)