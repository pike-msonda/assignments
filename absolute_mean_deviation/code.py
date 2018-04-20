import os
import time
import glob
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImageStat import Stat
from collections import OrderedDict
       

def normalise(values):
    x_max = 255
    x_min = 0
    norm = []
    for v in values:
        norm.append((v - x_min)/(x_max - x_min))
    return norm
        
def file_check(filename):
    """
    Checks and cleans the file to save the results to.
    """
    if(os.path.exists(filename)):
        os.remove(filename)
        f = create_empty_csv(filename)
        f.close()
        return
    else:
         create_empty_csv(filename)

def create_empty_csv(filename):
    """
    Create an empty csv file to be written to.
    """
    return open(filename, 'a')

def get_pixel_list(image):
    """
    Get full list of pixel values of an image
    """
    return list(image.getdata())

def stats(image):
    """
    Obtain basic image statics; mean of pixel values and total number of pixels
    """
    stat = Stat(image)
    return stat.mean, stat.count

def update_dictionary(dictionary, column, value):

    dict_to_update = {column:value}
    dictionary.update(dict_to_update)

def write_to_file(dictionary, filename):
    
    df = pd.DataFrame(dictionary, index = [0])

    try:
        if(os.stat(filename).st_size == 0):
            df.to_csv(filename, index=False, mode='a')

        else:
            df.to_csv(filename, index=False, mode='a',header=False)

    except Exception as e:
        print ("An Error occured while trying to write to file: " + e.message)
    
    #Writing in csv open file standard. Defualt seperation used 
    #df.to_csv("sample.csv", index=False, mode='a')

#TODO: Implement a more general MAD function
def mad(mean, pixel_list):
    """
        mean: mean of all the pixels on the image
        pixel_list: a list of all the pixels 

    Function to obtain Mean Absolute Deviation
    Written to work with only the current data items.
    """
    total = 0
    count = len(pixel_list)
    for p in pixel_list:
        abs_pixel =  np.abs(np.subtract(p, mean))
        total += abs_pixel

    #maintain the order of the results
    result  = np.divide(total, count)
    return result

#TODO: Streamline process using list comprehension
def splitImage(image,width,height,area):
    """
        Image: Image path to be processed.
        Width: width of sub image
        Height: height of sub image

    Split images based on a specified area. e.g: 20X20
    Standard format of images is :240X180
    """
    image_width, image_height = image.size
    sub_images = OrderedDict()
    count = 1

    #Iterate through the image, whilst creating croping boxes for 20x20 
    for i in range(0,image_height, height):
        for j in range(0, image_width, width):
            crop_area = (j, i, j + width, i + height )
            sub_image = image.crop(crop_area)
            sub_images.update({"SUB_IMAGE "+str(count): sub_image})
            count += 1

            try:
                #Check if croping was done.
                o =  sub_image.crop(area)
            except Exception:
                print("An error occurred when splitting the image" + o)
    return sub_images

if(__name__ == "__main__"):
    """
    Main program
    """
    start = time.time()

    #Name of output file.
    file_name = "results.csv"
    
    #Make sure the file is setup alright
    file_check(file_name) 
    image_name = "IMAGE " #class name
    image_serial_number = 1 

    #Iterate through all available images.
    for image in glob.glob("damar/*.jpg"):

        im = Image.open(image)
        converted_image = im
        split_images = splitImage(converted_image,20,20,(0,0,20,20))
        results = OrderedDict()
        results.update({"FILENAME :" : image_name+str(image_serial_number)})

        for name, image in split_images.items():
            pixel_list = get_pixel_list(image)
            mean, total = stats(image)
            normalised_mean = normalise(list(mean))
            normalised_pixel_list = normalise(pixel_list)
            res = mad(normalised_mean, normalised_pixel_list)
            update_dictionary(results, name, res)

        write_to_file(results,file_name)
        image_serial_number += 1
    
    print ("The program exectuted successfuly in: %s seconds" % (time.time() - start))
