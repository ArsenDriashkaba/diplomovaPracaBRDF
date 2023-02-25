package com.example.cameratestaplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.Button;
import android.widget.FrameLayout;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.util.Log;
import android.widget.ImageView;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Main Activity ------->";

    private static final int IMAGE_SIZE = 256;
    private static final int NUM_CHANNELS = 3;
    private static final String MODEL_PATH = "converted_model.tflite";
    private static final String IMAGE_PATH = "@drawable/test.png";

    FrameLayout frameLayout;
    Interpreter tfliteInterpreter;
    Bitmap bitmap;
    ImageView imageView;
    Button getResultBtn;


    @Override
    // Good
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        frameLayout = (FrameLayout)findViewById(R.id.frameLayout);

        getResultBtn = findViewById(R.id.button2);
        imageView = findViewById(R.id.photo);

        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);
        imageView.setImageBitmap(bitmap);

        handleLoadTfLite();
        handleLoadOpenCV();

        try {
            int[] outputShape = {1, IMAGE_SIZE, IMAGE_SIZE, 12};

            bitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);

            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(0f, 255f))
                            .build();

            // apply the ImageProcessor to the Bitmap to get a TensorImage
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmap);
            tensorImage = imageProcessor.process(tensorImage);

            // create a ByteBuffer to hold the tensor data
            TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(new int[]{1, 256, 256, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = tensorBuffer.getBuffer();

            //copy the image data to the buffer
            tensorImage.getBuffer().rewind();
            byteBuffer.rewind();
            byteBuffer.put(tensorImage.getBuffer());

            Log.e(TAG, Arrays.toString(tensorBuffer.getFloatArray()));

            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32);
            ByteBuffer outputData = outputBuffer.getBuffer();

            tfliteInterpreter.run(byteBuffer, outputData);

            //TODO: Data processing
            saveOutputTensorAsImage3channels(outputBuffer, "testImage.png", false);
        }catch(Exception ex){
            ex.printStackTrace();
        }

    }

    // Good
    private MappedByteBuffer loadModelfile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Good
    public void handleLoadTfLite(){
        try {
            tfliteInterpreter = new Interpreter(loadModelfile());

            Log.i(TAG, "TFLite model was succesfully loaded");
        } catch (Exception ex){
            Log.e(TAG, ex.toString());
        }
    }

    // Good
    public List<float[][][]> splitTensorBuffer(TensorBuffer inputBuffer, int width, int height) {
        /*
            Return list of 3-dims float array of size 256 x 256 x 3
            First is NORMAL_MAP
            Second is DIFFUSE
            Third is ROUGHNESS
            Last is SPECULAR
        */

        float[] outputArray = inputBuffer.getFloatArray();

        // Reshape the output array to a 4D tensor of shape (1, height, width, 12)
        int[] shape = inputBuffer.getShape();
        int numChannels = shape[3];
        int batchSize = shape[0];

        float[][][][] reshapedArray = new float[batchSize][height][width][numChannels];

        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    for (int c = 0; c < numChannels; c++) {
                        reshapedArray[b][i][j][c] = outputArray[b * height * width * numChannels + i * width * numChannels + j * numChannels + c];
                    }
                }
            }
        }

        // Split the 4D tensor into four 4D tensors, each of shape (1, height, width, 3)
        float[][][][] bitmap1Array = new float[batchSize][height][width][3];
        float[][][][] bitmap2Array = new float[batchSize][height][width][3];
        float[][][][] bitmap3Array = new float[batchSize][height][width][3];
        float[][][][] bitmap4Array = new float[batchSize][height][width][3];

        // Paste data into bitmapArrays
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    for (int c = 0; c < 3; c++) {
                        bitmap1Array[b][i][j][c] = reshapedArray[b][i][j][c];
                        bitmap2Array[b][i][j][c] = reshapedArray[b][i][j][c + 3];
                        bitmap3Array[b][i][j][c] = reshapedArray[b][i][j][c + 6];
                        bitmap4Array[b][i][j][c] = reshapedArray[b][i][j][c + 9];
                    }
                }
            }
        }

        List<float[][][]> listOfBDRFChannels = new ArrayList<>();

        listOfBDRFChannels.add(bitmap1Array[0]);
        listOfBDRFChannels.add(bitmap2Array[0]);
        listOfBDRFChannels.add(bitmap3Array[0]);
        listOfBDRFChannels.add(bitmap4Array[0]);

        return listOfBDRFChannels;
    }

    // Testing...
    public static Bitmap tensorBufferToImage(TensorBuffer tensorBuffer, boolean isRGB, boolean isSpecular) {
        float[] tensorArray = tensorBuffer.getFloatArray();

        float tensorMin = Float.MAX_VALUE;
        float tensorMax = Float.MIN_VALUE;

        // Find the minimum and maximum values in the tensor.
        for (float value : tensorArray) {
            if (value < tensorMin) {
                tensorMin = value;
            }
            if (value > tensorMax) {
                tensorMax = value;
            }
        }

        Bitmap bitmap;
        if (isSpecular) {
            // Multiply each value in the tensor by 255.
            for (int i = 0; i < tensorArray.length; i++) {
                tensorArray[i] *= 255f;
            }
            bitmap = Bitmap.createBitmap(tensorBuffer.getShape()[1], tensorBuffer.getShape()[0], Bitmap.Config.ALPHA_8);
        } else {
            // Scale the tensor values to the range [0, 255].
            for (int i = 0; i < tensorArray.length; i++) {
                tensorArray[i] = 255f * (tensorArray[i] - tensorMin) / (tensorMax - tensorMin);
            }
            bitmap = Bitmap.createBitmap(tensorBuffer.getShape()[1], tensorBuffer.getShape()[0], Bitmap.Config.ARGB_8888);
        }

        // Convert the float array to a byte array.
        byte[] byteArray = new byte[tensorArray.length * 4];
        ByteBuffer byteBuffer = ByteBuffer.wrap(byteArray);
        FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
        floatBuffer.put(tensorArray);

        if (isRGB) {
            // Convert the tensor to an RGB image.
            Mat mat = new Mat(tensorBuffer.getShape()[0], tensorBuffer.getShape()[1], CvType.CV_32FC3);
            mat.put(0, 0, byteBuffer.array());
            Mat rgbMat = new Mat();
            Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_BGR2RGB);
            Utils.matToBitmap(rgbMat, bitmap);
        } else {
            // Convert the tensor to a grayscale image.
            Mat mat = new Mat(tensorBuffer.getShape()[0], tensorBuffer.getShape()[1], CvType.CV_32FC1);
            mat.put(0, 0, byteBuffer.array());
            Utils.matToBitmap(mat, bitmap);
        }

        return bitmap;
    }

    // Testing...
    public Bitmap[] convertTensorBufferChannelsToBitmaps(TensorBuffer tensorBuffer, int width, int height) {
        List<float[][][]>listOfBDRFChannels = splitTensorBuffer(tensorBuffer, width, height);

        Log.e(TAG, Arrays.deepToString(listOfBDRFChannels.get(3)));

        // Convert each 4D tensor to a bitmap
        Bitmap[] bitmaps = new Bitmap[4];

        return bitmaps;
    }

    // Helper method to convert a 4D float array to a 1D integer array of colors
    private int[] convertArrayToColorInt(float[][][] array) {
        int[] intArray = new int[array.length * array[0].length * array[0][0].length];
        int index = 0;

        for (float[][] floats : array) {
            for (int j = 0; j < array[0].length; j++) {
                for (int k = 0; k < array[0][0].length; k++) {
                    int red = (int) (floats[j][k] * 255);
                    int green = (int) (floats[j][k] * 255);
                    int blue = (int) (floats[j][k] * 255);
                    int alpha = 255;
                    int color = (alpha << 24) + (red << 16) + (green << 8) + blue;
                    intArray[index++] = color;
                }
            }
        }
        return intArray;
    }

    // Testing...
    private void saveOutputTensorAsImage3channels(TensorBuffer outputTensor, String fileName, Boolean save) {
        Bitmap[] bitmaps = convertTensorBufferChannelsToBitmaps(outputTensor, IMAGE_SIZE, IMAGE_SIZE);
        Bitmap diffuse = bitmaps[0];

        imageView.setImageBitmap(diffuse);

        Bitmap outputBitmap = diffuse;

        if (save) {
            // Save the Bitmap as a PNG file
            try {
                File file = new File(getExternalFilesDir(null), fileName);
                FileOutputStream outputStream = new FileOutputStream(file);
                outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
                outputStream.close();
                Log.d(TAG, "Saved output image to " + file.getAbsolutePath());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // Good
    public void handleLoadOpenCV(){
        try{
            if (OpenCVLoader.initDebug()){
                Log.d(TAG, "Open CV loaded succesfully");
            }
        }catch (Exception ex){
            Log.e(TAG, ex.toString());
        }
    }
}