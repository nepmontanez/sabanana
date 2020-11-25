package com.appnana.sabanana;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Toast;

import com.appnana.sabanana.tflite.Classifier;
import com.appnana.sabanana.tflite.Recognition;
import com.appnana.sabanana.tflite.TFliteModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.model.Model;

import java.io.FileDescriptor;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private Intent requestFileIntent;
    private ParcelFileDescriptor inputPFD;
    private ImageView imageView;
    private Classifier model;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        this.model = new Classifier(this, "model.tflite", "labels.txt", TFliteModel.Device.CPU, 4);
        this.imageView = this.findViewById(R.id.imageView);
    }

    public void btnLoadImage(View view) {
        requestFileIntent = new Intent(Intent.ACTION_PICK);
        requestFileIntent.setType("image/jpg");
        this.startActivityForResult(requestFileIntent, 0);
    }

    public void btnPredictImage(View view) {
    }

    @SuppressLint("DefaultLocale")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent resultIntent) {
        if(resultCode != RESULT_OK){
            return;
        }
        else{
            // Get the file's content URI from the incoming Intent
            try {
                Uri returnUri = resultIntent.getData();
                inputPFD = getContentResolver().openFileDescriptor(returnUri, "r");
                FileDescriptor fd = inputPFD.getFileDescriptor();
                Bitmap image = BitmapFactory.decodeFileDescriptor(fd);
                List<Recognition> recognitions = model.runInference(image);
                final ListView list = findViewById(R.id.list);
                ArrayList<String> arrayList = new ArrayList<>();

                for(Recognition rec: recognitions){
                    arrayList.add(String.format("%s - %.2f%%", rec.getTitle(), (rec.getConfidence() * 100)));
                }
                ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, arrayList);
                list.setAdapter(arrayAdapter);
                list.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                    @Override
                    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                        String clickedItem=(String) list.getItemAtPosition(position);
//                        Toast.makeText(MainActivity.this,clickedItem,Toast.LENGTH_LONG).show();
                    }
                });
                Log.d("DEBUG", String.valueOf(image.getHeight()));
                this.imageView.setImageBitmap(image);
                inputPFD.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
        super.onActivityResult(requestCode, resultCode, resultIntent);
    }

}

