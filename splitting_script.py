import splitfolders
'''
input_folder_h = r'C:/Dane/for_split1/healthy'
input_folder_t = r'C:/Dane/for_split1/tumor'

output_folder_h = r'C:/Dane/for_split/healthy'
output_folder_t = r'C:/Dane/for_split/tumor'

#output_folder = r'C:/Dane/40'
'''

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#Train, val, test
ratio = 0.4
input_folder = r'C:/Dane/for_split'
output_folder = r'C:/Dane/balanced_kfold_test'

splitfolders.fixed(input=input_folder, output=output_folder,
                   seed=1334, fixed=(503, 50),
                   group_prefix=None) # default values




'''
from PIL import Image
import os

#tumor_dir = r'C:/Brain_Tumor_Dataset/Tumor'
#healthy_dir = r'C:/Brain_Tumor_Dataset/Healthy'

#tumor_dir_jpg = r'C:/Dane/Brain_Tumor_JPG/tumor'
#healthy_dir_jpg = r'C:/Dane/Brain_Tumor_JPG/healthy'

#new_h = r'C:/Dane/100/healthy'
#new_t = r'C:/Dane/100/tumor'


i = 1
for filename in os.listdir(input_folder_h):
    im = Image.open(os.path.join(input_folder_h, filename))
    im = im.convert('RGB')

    new_name = "Healthy_" + str(i) + ".jpg"
    im.save(os.path.join(output_folder_h, new_name))

    i += 1

i = 1
for filename in os.listdir(input_folder_t):
    im = Image.open(os.path.join(input_folder_t, filename))
    im = im.convert('RGB')

    new_name = "Tumor_" + str(i) + ".jpg"
    im.save(os.path.join(output_folder_t, new_name)) 
    
    i += 1 
    
'''