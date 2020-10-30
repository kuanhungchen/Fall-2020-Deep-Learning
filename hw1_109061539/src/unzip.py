import cv2
import gzip
import struct
import os
import numpy as np


def unzip_images(src_filename, dst_filename):
    # read content in zip file
    with gzip.open(src_filename) as zipfile:
        _ = struct.unpack('I', zipfile.read(4))
        num_of_data = struct.unpack('>I', zipfile.read(4))[0]
        print("number of data: {}".format(num_of_data))

        r = struct.unpack('>I', zipfile.read(4))[0]
        c = struct.unpack('>I', zipfile.read(4))[0]
        print("shape of image: {} x {}".format(r, c))

        data = np.fromstring(zipfile.read(num_of_data * r * c), dtype=np.uint8)
        data = data.reshape((num_of_data, r, c))
    zipfile.close()
    
    # save images
    for i in range(num_of_data):
        cv2.imwrite(os.path.join(dst_filename, str(i).zfill(5) + ".png"), data[i])

def unzip_labels(src_filename, dst_filename):
    # read content in zip file
    with gzip.open(src_filename) as zipfile:
        _ = struct.unpack('I', zipfile.read(4))
        num_of_data = struct.unpack('>I', zipfile.read(4))[0]
        print("number of data: {}".format(num_of_data))
        
        data = np.fromstring(zipfile.read(num_of_data), dtype=np.uint8)
        data = data.reshape((num_of_data, 1))
    zipfile.close()
   
    # save labels
    for i in range(num_of_data):
        with open(os.path.join(dst_filename, str(i).zfill(5) + ".txt"), "w") as fp:
            fp.write(str(data[i]))
        fp.close()

if __name__ == "__main__":
    src_path = "MNIST/zip_files/train-images.gz"
    dst_path = "MNIST/train/images/"
    unzip_images(src_path, dst_path)


# train_images_filename = "data/train/images.gz"
# with gzip.open(train_images_filename) as tr_img:
    # n = struct.unpack('>I', tr_img.read(4))[0]
    # img_num = struct.unpack('>I', tr_img.read(4))[0]
    # res = np.fromstring(tr_img.read(60000 * 28 * 28), dtype=np.uint8)
# tr_img.close()
# res = res.reshape((60000, (28 * 28)))

# print(res.shape)


