import io
import glob
import os.path

for i, file in enumerate(glob.glob('data/strokes_background/*/*/*txt')):
    file_id = os.path.basename(file).strip().strip('.txt')
    img_file = f'data/images/train/{file_id}.png'
    if (i + 1) % 500 == 0:
        print(f"File no {i + 1}")
    if not os.path.isfile(img_file):
        print("file_id {file_id} no img file!")
    file = io.open(file, 'r')
    lines = [line.strip() for line in file]
    file.close()
    if 'START' not in lines:
        print(f"No start in {file_id}")
    if 'BREAK' not in lines:
        print(f"No break in {file_id}")
    if lines[-1] != 'BREAK':
        print(f"Break is not the end in {file_id}")
