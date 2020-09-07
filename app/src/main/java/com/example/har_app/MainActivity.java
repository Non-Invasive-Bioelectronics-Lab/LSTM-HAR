package com.example.har_app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;



//Data is streamed into the main activity and the predictions are output to the UI
public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final String TAG  = "";
    private final int N_Samples = 128;
    /*private static List<Float> ax, ay, az;
    private static List<Float> lx, ly, lz;
    private static List<Float> gx, gy, gz;*/ //For embedded sensor implementation
    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mLinearAccelerometer;
    private Sensor mGyroscope;
    private float[][] results;
    private HARClassifier classifier;

    private TextView walkingTextView;
    private TextView upstairsTextView;
    private TextView downstairsTextView;
    private TextView sittingTextView;
    private TextView standingTextView;
    private TextView lyingTextView;

    // Load input test data files
    private static final String ax_file = "total_acc_x_test.txt";
    private static final String ay_file = "total_acc_y_test.txt";
    private static final String az_file = "total_acc_z_test.txt";
    private static final String lx_file = "body_acc_x_test.txt";
    private static final String ly_file = "body_acc_y_test.txt";
    private static final String lz_file = "body_acc_z_test.txt";
    private static final String gx_file = "body_gyro_x_test.txt";
    private static final String gy_file = "body_gyro_y_test.txt";
    private static final String gz_file = "body_gyro_z_test.txt";

    private static float[][] ax, ay, az;
    private static float[][] lx, ly, lz;
    private static float[][] gx, gy, gz;

    private int i = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        walkingTextView = findViewById(R.id.walkingTextView);
        upstairsTextView = findViewById(R.id.upstairsTextView);
        downstairsTextView = findViewById(R.id.downstairsTextView);
        sittingTextView = findViewById(R.id.sittingTextView);
        standingTextView = findViewById(R.id.standingTextView);
        lyingTextView = findViewById(R.id.lyingTextView);

        /*ax = new ArrayList<>();
        ay = new ArrayList<>();
        az = new ArrayList<>();
        lx = new ArrayList<>();
        ly = new ArrayList<>();
        lz = new ArrayList<>();
        gx = new ArrayList<>();
        gy = new ArrayList<>();
        gz = new ArrayList<>();*/ //For embedded sensor implementation

        //Load required sensors, if used
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);

        mLinearAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mSensorManager.registerListener(this, mLinearAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);

        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);

        //Load pre-collected Test Data into 9 individual channels for easier handling; this is only done when using pre-collected data
        try{
            classifier = new HARClassifier(getApplicationContext()); //try/catch?
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }
        try {
            ax = classifier.LoadTest(getApplicationContext(), ax_file);
            ay = classifier.LoadTest(getApplicationContext(), ay_file);
            az = classifier.LoadTest(getApplicationContext(), az_file);
            lx = classifier.LoadTest(getApplicationContext(), lx_file);
            ly = classifier.LoadTest(getApplicationContext(), ly_file);
            lz = classifier.LoadTest(getApplicationContext(), lz_file);
            gx = classifier.LoadTest(getApplicationContext(), gx_file);
            gy = classifier.LoadTest(getApplicationContext(), gy_file);
            gz = classifier.LoadTest(getApplicationContext(), gz_file);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Stream pre-collected Test Data using a timer to simulate real-time data collection and inference
        final Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (i == 2947) { //If last element of dataset has been input
                    Log.i(TAG,"Dataset is finished, interpreter is closed, timer is stopped");
                    classifier.interpreter.close(); //Close interpreter when finished
                    timer.cancel();
                    return;
                }
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (i < 2947) {
                            HARPrediction(i); //Make prediction on data input for single window
                            i++;

                        }
                    }
                });
                //i++;
            }
        },0,500); //Sets delay and period of timer to stream data in ms (one window of data is 2.56s, suggest to decrease for faster testing)


    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mLinearAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mSensorManager.unregisterListener(this);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
       /* If Embedded Sensors Used
        HARPrediction();
        Sensor sensor = event.sensor;
        if(sensor.getType() == Sensor.TYPE_ACCELEROMETER && ax.size() < N_Samples ){
            ax.add(event.values[0]);
            ay.add(event.values[1]);
            az.add(event.values[2]);
        } else if(sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION && lx.size() < N_Samples){
            lx.add(event.values[0]);
            ly.add(event.values[1]);
            lz.add(event.values[2]);
        } else if(sensor.getType() == Sensor.TYPE_GYROSCOPE && gx.size() < N_Samples) {
            gx.add(event.values[0]);
            gy.add(event.values[1]);
            gz.add(event.values[2]);
        }*/

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }


    private void HARPrediction(int i){
        //List<Float> data = new ArrayList<>(); //Uncomment if embedded sensors used

        /*if(ax.size() == N_Samples && ay.size() == N_Samples && az.size() == N_Samples
                && lx.size() == N_Samples && ly.size() == N_Samples && lz.size() == N_Samples
                && gx.size() == N_Samples && gy.size() == N_Samples && gz.size() == N_Samples) {*/

        //Load parameters for MinMax scaling of each input channel
        float ax_max = 2.197618f; float ax_min = -0.4665558f; float ax_dlt = ax_max - ax_min;
        float ay_max = 1.21735f; float ay_min = -1.582079f; float ay_dlt = ay_max - ay_min;
        float az_max = 1.281363f; float az_min = -1.639609f; float az_dlt = az_max - az_min;

        float lx_max = 1.299120f; float lx_min = -1.232238f; float lx_dlt = lx_max - lx_min;
        float ly_max = 0.9759764f; float ly_min = -1.345267f; float ly_dlt = ly_max - ly_min;
        float lz_max = 1.066916f; float lz_min = -1.364707f; float lz_dlt = lz_max - lz_min;

        float gx_max = 4.155473f; float gx_min = -4.733656f; float gx_dlt = gx_max - gx_min;
        float gy_max = 5.746062f; float gy_min = -5.97433f; float gy_dlt = gy_max - gy_min;
        float gz_max = 2.365982f; float gz_min = -2.763014f; float gz_dlt = gz_max - gz_min;

        //Segment and Reshape Data into fixed window sizes
        float[][][] input_3d = new float[1][128][9];
        for (int n = 0; n < 128; n++) {

            input_3d[0][n][0] = (2*((ax[i][n]) - ax_min)/ax_dlt) - 1;
            input_3d[0][n][1] = (2*((ay[i][n]) - ay_min)/ay_dlt) - 1;
            input_3d[0][n][2] = (2*((az[i][n]) - az_min)/az_dlt) - 1;

            input_3d[0][n][3] = (2*((lx[i][n]) - lx_min)/lx_dlt) - 1;
            input_3d[0][n][4] = (2*((ly[i][n]) - ly_min)/ly_dlt) - 1;
            input_3d[0][n][5] = (2*((lz[i][n]) - lz_min)/lz_dlt) - 1;

            input_3d[0][n][6] = (2*((gx[i][n]) - gx_min)/gx_dlt) - 1;
            input_3d[0][n][7] = (2*((gy[i][n]) - gy_min)/gy_dlt) - 1;
            input_3d[0][n][8] = (2*((gz[i][n]) - gz_min)/gz_dlt) - 1;

        }

        //Make predictions on input data window in HAR Classifier
        results = classifier.predictions(input_3d);

        //Output predictions to app UI
        walkingTextView.setText("Walking: \t" + round(results[0][0], 2));
        upstairsTextView.setText("Upstairs: \t" + round(results[0][1], 2));
        downstairsTextView.setText("Downstairs: \t" + round(results[0][2], 2));
        sittingTextView.setText("Sitting: \t" + round(results[0][3], 2));
        standingTextView.setText("Standing: \t" + round(results[0][4], 2));
        lyingTextView.setText("Lying: \t" + round(results[0][5], 2));

        //Uncomment to clear data if embedded sensors used
        /*data.clear();
        ax.clear(); ay.clear(); az.clear();
        lx.clear(); ly.clear(); lz.clear();
        gx.clear(); gy.clear(); gz.clear();*/

    }

    /* Uncomment if embedded sensors used, as data needs to be converted to a float array from a Float ArrayList
    private float[] toFloatArray(List<Float> list){
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list){
            array[i++] = (f != null ? f :Float.NaN);
        }
        return array;
    }*/

    //Rounds the output predictions to two decimal places
    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

}