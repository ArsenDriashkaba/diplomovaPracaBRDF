import numpy as np
import tensorflow as tf
import PIL
import math
from PIL import Image

def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

def activate(tensor):
    s = np.vectorize(sigmoid)

    return s(tensor)

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

def gamma_correct(gamma, folder, imgSrc):
    im = Image.open(f'{folder}/{imgSrc}')
    gamma1 = gamma
    row = im.size[0]
    col = im.size[1]
    result_img1 = Image.new(mode="RGB", size=(row, col), color=0)
    for x in range(row):
        for y in range(col):
            r = pow(im.getpixel((x, y))[0] / 255, (1 / gamma1)) * 255
            g = pow(im.getpixel((x, y))[1] / 255, (1 / gamma1)) * 255
            b = pow(im.getpixel((x, y))[2] / 255, (1 / gamma1)) * 255
            # add
            color = (int(r), int(g), int(b))
            result_img1.putpixel((x, y), color)
    #show
    result_img1.save(f'{folder}/corrected-{imgSrc}')
    result_img1.show()

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

imgSrc = 'IMG_20180105_091814.png'

gamma_correct(2, 'inputExamples', imgSrc)

image_string = tf.io.read_file(f'inputExamples/corrected-{imgSrc}')
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


#_________________________________Specular (9:12)
outputed1 = output_data[0,:,:,9:12]
outputedRGB1 = tensor_to_image(outputed1, False, True)

outputedRGB1.save('tflite_images/my8.png')

im1 = Image.open('tflite_images/my8.png')

#__________________________________Roughness
outputed2 = output_data[0,:,:,6:9]
outputedRGB2 = tensor_to_image(outputed2)

outputedRGB2.save('tflite_images/my9.png')

im2 = Image.open('tflite_images/my9.png')


#__________________________________Normals
outputed3 = output_data[0,:,:,0:3]
outputedRGB2 = tensor_to_image(outputed3, True)

outputedRGB2.save('tflite_images/my10.png')

im3 = Image.open('tflite_images/my10.png')



im_final = get_concat_h(get_concat_h(get_concat_h(im, im1), im2), im3)

im_final.show()