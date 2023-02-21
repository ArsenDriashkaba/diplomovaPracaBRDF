package com.example.cameratestaplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.Button;
import android.widget.FrameLayout;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import org.opencv.android.OpenCVLoader;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
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
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        frameLayout = (FrameLayout)findViewById(R.id.frameLayout);

        getResultBtn = findViewById(R.id.button2);
        imageView = findViewById(R.id.photo);

        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);
        imageView.setImageBitmap(bitmap);

        handleLoadTfLite();

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                        .build();

        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);
        tensorImage = imageProcessor.process(tensorImage);

        ByteBuffer inputBuffer = tensorImage.getBuffer();

        try {
            int[] outputShape = {1, bitmap.getHeight(), bitmap.getWidth(), 12};
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32);
            tfliteInterpreter.run(inputBuffer, outputBuffer.getBuffer());

            Log.d(TAG, "Yoohooooooooo");

            //TODO: Data processing

        }catch(Exception ex){
            Log.e(TAG, ex.toString());
        }

    }

    private MappedByteBuffer loadModelfile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void handleLoadTfLite(){
        try {
            tfliteInterpreter = new Interpreter(loadModelfile());

            Log.i(TAG, "TFLite model was succesfully loaded");
        } catch (Exception ex){
            Log.e(TAG, ex.toString());
        }
    }


    public void handleLoadOpenCV(){
        try{
            if (OpenCVLoader.initDebug()){
                Log.d(TAG, "Open CV loaded succesfully");
            }
        }catch (Exception ex){
            Log.e(TAG, ex.toString());
        }
    }


    public int[] convertBitmapToIntArray(Bitmap bitmap){
        int x = bitmap.getWidth();
        int y = bitmap.getHeight();
        int[] intArray = new int[x * y];

        bitmap.getPixels(intArray, 0, x, 0, 0, x, y);

        return intArray;
    }
}