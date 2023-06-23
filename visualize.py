# C:\Users\zy7\AppData\Local\Temp\4\openai-2023-06-20-16-46-29-036293\samples_100x128x128x3.npz

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


#data_path = r'C:\Users\zy7\AppData\Local\Temp\4\openai-2023-06-20-16-46-29-036293\samples_100x128x128x3.npz'
#data_path = r'C:\Users\zy7\AppData\Local\Temp\4\openai-2023-06-21-14-55-04-687904\samples_2x512x512x3.npz'
#data_path = r'C:\Users\zy7\AppData\Local\Temp\4\openai-2023-06-21-15-10-45-305644\samples_2x512x512x3.npz'
data_path = r'C:\Users\zy7\AppData\Local\Temp\4\openai-2023-06-22-12-47-26-094243\samples_2x128x128x3.npz'
# Create a folder to save the PNG files
output_folder = 'sample_output/512conditional_2'
os.makedirs(output_folder, exist_ok=True)

a = np.load(data_path)['arr_0']
print('shape a = ', np.shape(a))
b = a[0]

print('shape b = ', np.shape(b))
plt.imshow(b)
plt.show()

for i, image in enumerate(a):
    image_path = os.path.join(output_folder, f'image_{i}.png')
    pil_image = Image.fromarray(image)
    pil_image.save(image_path)

print("Images saved successfully.")
#b = b/np.max(b)*255
#b = np.array(b, dtype = np.uint8) 

