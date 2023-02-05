import numpy as np
import os
import tensorflow as tf
import PIL
from PIL import Image
from matplotlib import cm
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--mode", required=True, choices=["test", "train", "eval"])
# parser.add_argument("--output_dir", required=True, help="where to put output files")
# parser.add_argument("--input_dir", help="path to xml file, folder or image (defined by --imageFormat) containing information images")
# parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

# a = parser.parse_args()

def tensor_to_image(tensor1, isRGB = False):
    tensor = tensor1*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    if isRGB:
        PIL.Image.fromarray(tensor, 'RGB')
    return PIL.Image.fromarray(tensor)

def otherFunction(tensor):
    t = tensor*255
    t = np.array(tensor, dtype=np.uint8)
    return PIL.Image.fromarray(np.uint8(cm.gist_earth(t)))

def tensor_to_BRDF(tensor):
    partialOutputedNormals = tensor[:,:,:,0:2]
    outputedDiffuse = tensor[:,:,:,2:5]
    outputedRoughness = tensor[:,:,:,5]
    outputedSpecular = tensor[:,:,:,6:9]

    return tensor_to_image(outputedDiffuse)


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test photo load to tflite model
# examples = load_examples(a.input_dir, a.mode == "train")
# inputs = deprocess(examples.inputs)
# inputs_reshaped = reshape_tensor_display(inputs, 1, logAlbedo = False)
# converted_inputs = convert(inputs_reshaped)
# real_inputs = tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs")
# print(real_inputs[0])
                                       
# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

image_string = tf.io.read_file('inputExamples/IMG_20180115_143924.png')
raw_input = tf.image.decode_image(image_string)
raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

raw_input1 = np.array([raw_input])

interpreter.set_tensor(input_details[0]['index'], raw_input1)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

# tensor_to_BRDF(output_data)

outputed = output_data[:,:,:,3:6]
outputedRGB = tensor_to_image(outputed[0])

outputedRGB.save('test_images/my7.png')

im = Image.open('test_images/my7.png')
im.show()
