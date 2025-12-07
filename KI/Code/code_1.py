import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

with h5py.File("100.mat", "r") as f:
   cj = f['cjdata']
   print(cj.keys())

   #Patientenid 
   pid = np.array(cj["PID"])
   # ASCII in Strings 
   pid_str = "".join(chr(c[0]) for c in pid)
   print(pid_str)

    #Bilder  
   image = np.array(cj['image'])
   plt.imshow(image, cmap="gray")
   plt.axis("off")
   plt.show()
   print(image)


   #Label 
   label = np.array(cj['label'])
   print(label)
   if label == 1: 
      print('m')
   elif label == 2: 
      print('m')

   #Tumorborder 
   border = np.array(cj['tumorBorder'])
   print(border)

   #Maske 
   maske = np.array(cj['tumorMask'])
   plt.imshow(image, cmap = 'grey')
   plt.imshow(maske, cmap = 'Reds', alpha = 0.4)
   plt.show()

  
