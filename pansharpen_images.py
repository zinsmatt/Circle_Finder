import os
import glob
import subprocess



dataset_folder = "/media/DATA1/Topcoder/circle_finder/train/*"

folders = glob.glob(dataset_folder)

for f in folders:    
    pan_file = os.path.join(f, os.path.basename(f) + "_PAN.tif")
    mul_file = os.path.join(f, os.path.basename(f) + "_MUL.tif")
    out_file = os.path.join(f, os.path.basename(f) + "_PANSHARPEN.tif")
    
    cmd = ["gdal_pansharpen.py", pan_file, mul_file, out_file, "-b", "3", "-b", "2", "-b", "1"]
    print(out_file)
    subprocess.run(cmd)
    

