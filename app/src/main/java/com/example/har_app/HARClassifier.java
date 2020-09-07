package com.example.har_app;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.util.Arrays;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.Tensor;

//This classifier performs HAR inference on the data input into the Main Activity
public class HARClassifier {
    //Load instance of interpreter from TFlite Android support library
    protected Interpreter interpreter;
    private static final String Model_Name = "LSTM_float.tflite"; //Load model for testing (float-32 or unit-8)
    public int[] output_shape;
    public int output_index = 0;

    //Load Model into Interpreter
    public HARClassifier(Context context) throws IOException {
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(context, Model_Name);
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        interpreter = new Interpreter(tfliteModel, tfliteOptions);

    }

    public float[][] predictions(float[][][] input_data){
        float[][] results = new float[1][6]; //Output shape of results
        interpreter.run(input_data,results); //Run Interpreter on data input from Main Activity
        output_shape = interpreter.getOutputTensor(output_index).shape(); //Output tensor shape
        int[] input_ten = interpreter.getInputTensor(output_index).shape(); //Input tensor shape


        return results; //Return prediction results
    }
    public float[][] LoadTest(Context context, String file) throws IOException {
        //Read input data and divide into 9 individuals channels of 128x9 arrays
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(file)));
        int rows = 2947;//2947
        int columns = 128;//128

        String[] line;
        float[][] myArray = new float[rows][columns];

        for (int i = 0; i < rows; i++) {
            line = reader.readLine().trim().split("\\s+");
            if (line == null) {
                reader.close();
            }
            for (int j = 0; j < columns; j++) {
                float k = Float.parseFloat(line[j]);
                myArray[i][j] = k;
            }

        }

        return myArray;
    }
}
