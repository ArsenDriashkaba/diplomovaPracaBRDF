package com.example.cameratestaplication;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;

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
import org.opencv.core.Core;
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
import android.widget.RelativeLayout;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Main Activity ------->";
    private static final int CAMERA_REQUEST_CODE = 1111;
    private static final int IMAGE_CAPTURED_CODE = 1112;
    private static final int IMAGE_SELECT_CODE = 1113;

    private static final int IMAGE_SIZE = 256;
    private static final int NUM_CHANNELS = 3;
    private static final String MODEL_PATH = "converted_model.tflite";
    private static final String IMAGE_PATH = "tree1.png";

    RelativeLayout imagesLayout;
    Interpreter tfliteInterpreter;
    Bitmap bitmap, photoBitmap;
    ImageView normalView, diffuseView, roughnessView, specularView, actualPhoto;
    Button getResultBtn, captureBtn, galleryBtn;


    @Override
    // Good
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imagesLayout = (RelativeLayout)findViewById(R.id.imagesLayout);

        getResultBtn = findViewById(R.id.button2);
        captureBtn = findViewById(R.id.button);
        galleryBtn = findViewById(R.id.gallery);

        normalView = findViewById(R.id.photo);
        diffuseView = findViewById(R.id.photo1);
        roughnessView = findViewById(R.id.photo2);
        specularView = findViewById(R.id.photo3);
        actualPhoto = findViewById(R.id.photo4);

        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.tree2);

        handleLoadTfLite();
        handleLoadOpenCV();
        getPermission();

        // Pass bitmap to process image and display it on screen
        processImage(bitmap);

        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, IMAGE_CAPTURED_CODE);
            }
        });

        galleryBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("images/*");
                startActivityForResult(intent, IMAGE_SELECT_CODE);
            }
        });

    }

    // Good
    public void processImage(Bitmap bitmap){
        try {
            int[] outputShape = {1, IMAGE_SIZE, IMAGE_SIZE, 12};

            bitmap = cropBitmapToSquare(bitmap);

            // Testing...
            bitmap = gammaCorrectBitmap(bitmap, 2f);
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

//            Log.e(TAG, Arrays.toString(tensorBuffer.getFloatArray()));

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

    // Testing...
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
                        // Add scaling here if everything will work
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

    // Good
    public Bitmap[] convertTensorBufferChannelsToBitmaps(TensorBuffer tensorBuffer, int width, int height) {
        List<float[][][]>listOfBDRFChannels = splitTensorBuffer(tensorBuffer, width, height);

        // Convert each 4D tensor to a bitmap
        Bitmap[] bitmaps = new Bitmap[4];

        Bitmap normal = convertFloatArrayToBitmap(listOfBDRFChannels.get(0), 1);
        Bitmap diffuse = convertFloatArrayToBitmap(listOfBDRFChannels.get(1), 0.55f);
        Bitmap roughness = convertFloatArrayToBitmap(listOfBDRFChannels.get(2), 1.5f);
        Bitmap specular = convertFloatArrayToBitmap(listOfBDRFChannels.get(3), 1.5f);

        bitmaps[0] = normal;
        bitmaps[1] = diffuse;
        bitmaps[2] = roughness;
        bitmaps[3] = specular;

        return bitmaps;
    }

    // Testing...
    public Bitmap convertFloatArrayToBitmap(float[][][] floatArray, float gamma) {
        int rows = floatArray.length;
        int cols = floatArray[0].length;
        Mat mat = new Mat(rows, cols, CvType.CV_32FC3);

        float tMin = Float.MAX_VALUE;
        float tMax = Float.MIN_VALUE;

        // Find the minimum and maximum values in the tensor.
        for (float[][] value : floatArray) {
            for (float[] value1: value){
                for (float v : value1){
                    if (v < tMin) {
                        tMin = v;
                    }
                    if (v > tMax) {
                        tMax = v;
                    }
                }
            }
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float[] pixelValues = new float[3];
                pixelValues[0] = floatArray[i][j][0];
                pixelValues[1] = floatArray[i][j][1];
                pixelValues[2] = floatArray[i][j][2];

                float[] scaledArray = scaleArrayToRGBValues(pixelValues, false, tMin, tMax);

                mat.put(i, j, scaledArray);
            }
        }

        Mat convertedMat = new Mat();
        mat.convertTo(convertedMat, CvType.CV_8UC3);

        if (gamma != 1){
            // Testing...
            convertedMat = gammaCorrection(convertedMat, gamma);
        }

        Bitmap bitmap = Bitmap.createBitmap(cols, rows, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(convertedMat, bitmap);

        return bitmap;
    }

    // Good
    public float[] scaleArrayToRGBValues(float[] array, boolean isSpecular, float tMin, float tMax) {
        float[] scaledArray = new float[array.length];
        float scale = 255.0f;

        if (isSpecular) {
            // Multiply each value in the tensor by 255.
            for (int i = 0; i < array.length; i++) {
                scaledArray[i] = Math.round(array[i] * scale);
            }

            return scaledArray;
        }

        // Scale the tensor values to the range [0, 255].
        for (int i = 0; i < array.length; i++) {
            scaledArray[i] = Math.round(scale * (array[i] - tMin) / (tMax - tMin));
        }

        return scaledArray;
    }

    // Testing...
    private void saveOutputTensorAsImage3channels(TensorBuffer outputTensor, String fileName, Boolean save) {
        Bitmap[] bitmaps = convertTensorBufferChannelsToBitmaps(outputTensor, IMAGE_SIZE, IMAGE_SIZE);
        Bitmap diffuse = bitmaps[0];

        normalView.setImageBitmap(bitmaps[0]);
        diffuseView.setImageBitmap(bitmaps[1]);
        roughnessView.setImageBitmap(bitmaps[2]);
        specularView.setImageBitmap(bitmaps[3]);

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
    public static Bitmap cropBitmapToSquare(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        if (width == height){
            return bitmap;
        }

        if (height > width) {
            return Bitmap.createBitmap(bitmap, 0, height / 2 - width / 2,
                    width, width);
        }

        return Bitmap.createBitmap(bitmap, width / 2 - height / 2, 0,
                    height, height);
    }

    // Good
    public static Mat gammaCorrection(Mat src, float gamma) {
        float invGamma = 1 / gamma;

        Mat table = new Mat(1, 256, CvType.CV_8U);
        Mat gammaCorrected = new Mat();

        for (int i = 0; i < 256; ++i) {
            table.put(0, i, (int) (Math.pow(i / 255.0f, invGamma) * 255));
        }

        Core.LUT(src, table, gammaCorrected);

        return gammaCorrected;
    }

    // Good
    public static Bitmap gammaCorrectBitmap(Bitmap src, float gamma) {
        // Convert bitmap to Mat object
        Mat matImage = new Mat();
        Utils.bitmapToMat(src, matImage);

        // Apply gamma correction
        matImage = gammaCorrection(matImage, gamma);

        // Convert back to bitmap
        Mat convertedMat = new Mat();
        matImage.convertTo(convertedMat, CvType.CV_8UC3);

        Bitmap bitmap = Bitmap.createBitmap(convertedMat.rows(), convertedMat.cols(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(convertedMat, bitmap);

        return bitmap;
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

    // Good
    void getPermission(){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
            }
        }
    }

    // Good
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == CAMERA_REQUEST_CODE){
            if (grantResults.length > 0){
                if (grantResults[0] != PackageManager.PERMISSION_GRANTED){
                    this.getPermission();
                }
            }
        }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    // Good
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        try {
            if (requestCode == IMAGE_CAPTURED_CODE && data != null){
                photoBitmap = (Bitmap) data.getExtras().get("data");

                new Thread(new Runnable() {
                    public void run(){
                        // Testing processing some texture:
                        processImage(photoBitmap);
                    }
                }).start();

                actualPhoto.setImageBitmap(photoBitmap);

                return;
            }

            if (requestCode == IMAGE_SELECT_CODE && data != null){
                Uri uri = data.getData();
                photoBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);

                new Thread(new Runnable() {
                    public void run(){
                        processImage(photoBitmap);
                    }
                }).start();

                actualPhoto.setImageBitmap(photoBitmap);

                return;
            }
        }catch (Exception ex){
            ex.printStackTrace();
        }

        super.onActivityResult(requestCode, resultCode, data);
    }
}