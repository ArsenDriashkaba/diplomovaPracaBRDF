import numpy as np
import os
import tensorflow as tf
import PIL
from PIL import Image
from matplotlib import cm

def tensor_to_image(tensor1, isRGB = False, isSpecular=False):
    t = tensor1
    t_min = np.min(t)
    t_max = np.max(t)

    if isSpecular:
        tensor = 255 * t
    else:
        tensor = 255 * (t - t_min) / (t_max - t_min)
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    if isRGB:
        PIL.Image.fromarray(tensor, 'RGB')
    return PIL.Image.fromarray(tensor)

def otherFunction(tensor):
    t = tensor

    print(t[0])

    return PIL.Image.fromarray(tensor, 'RGB')

def tensor_to_BRDF(tensor):
    partialOutputedNormals = tensor[:,:,:,0:2]
    outputedDiffuse = tensor[:,:,:,2:5]
    outputedRoughness = tensor[:,:,:,5]
    outputedSpecular = tensor[:,:,:,6:9]

    return tensor_to_image(outputedDiffuse)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

image_string = tf.io.read_file('inputExamples/IMG_6966.png')
raw_input = tf.image.decode_image(image_string)
raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

raw_input1 = np.array([raw_input])

interpreter.set_tensor(input_details[0]['index'], raw_input1)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

# _________________________________Diffuse (3:6)
outputed = output_data[0,:,:,3:6]
outputedRGB = tensor_to_image(outputed, True)

outputedRGB.save('tflite_images/my7.png')

im = Image.open('tflite_images/my7.png')

# im.show()


#_________________________________Specular (9:12)
outputed1 = output_data[0,:,:,9:12]
outputedRGB1 = tensor_to_image(outputed1, False, True)

outputedRGB1.save('tflite_images/my8.png')

im1 = Image.open('tflite_images/my8.png')

# im1.show()

#__________________________________Roughness
outputed2 = output_data[0,:,:,6:9]
outputedRGB2 = tensor_to_image(outputed2)

outputedRGB2.save('tflite_images/my9.png')

im2 = Image.open('tflite_images/my9.png')

# im2.show()

#__________________________________Normals
outputed3 = output_data[0,:,:,0:3]
outputedRGB2 = tensor_to_image(outputed3, True)

outputedRGB2.save('tflite_images/my10.png')

im3 = Image.open('tflite_images/my10.png')

# im3.show()

im_final = get_concat_h(get_concat_h(get_concat_h(im, im1), im2), im3)

im_final.show()
