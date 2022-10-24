import glob
from PIL import Image
import os
def make_gif(frame_folder, letter):
    files=glob.glob(f"{frame_folder}/{letter}*0.jpg")
    filesSort = [int(filed.split('_')[-1].split('.')[0]) for filed in files]
    print(filesSort)
    files = [x for _,x in sorted(zip(filesSort, files))]
    frames = [Image.open(image) for image in files]
    print(files)
    frame_one = frames[0]
    frame_one.save(f"/data/vision/polina/users/aunell/mri-langevin/csgm-mri-langevin/outputs/{letter}.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
print("testing")
make_gif("/data/vision/polina/users/aunell/mri-langevin/csgm-mri-langevin/outputs/2022-10-21/16-32-34", "f")
#make_gif("/data/vision/polina/users/aunell/mri-langevin/csgm-mri-langevin/outputs/2022-10-17/09-59-33", "P")
