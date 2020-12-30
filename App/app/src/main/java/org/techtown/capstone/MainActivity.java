package org.techtown.capstone;

import android.content.Context;
import android.content.Intent;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.maps.CameraUpdate;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.MapsInitializer;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.pedro.library.AutoPermissions;
import com.pedro.library.AutoPermissionsListener;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.converter.scalars.ScalarsConverterFactory;


public class MainActivity extends AppCompatActivity implements AutoPermissionsListener {
    //애뮬레이터용: http://10.0.2.2:8000/
    //apk용: http://192.168.25.3:8000/ - 우리집 기준
    Retrofit retrofit = new Retrofit.Builder()
            .baseUrl("http://10.0.2.2:8000/")
            .addConverterFactory(ScalarsConverterFactory.create())
            .addConverterFactory(GsonConverterFactory.create())
            .build();

    SupportMapFragment mapFragment;
    GoogleMap map;

    MarkerOptions myLocationMarker;
    int val = -1;

    public double latitude = 0;
    public double longitude = 0;
    public int nowWeather = 0;
    public String nowLocal = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        mapFragment = (SupportMapFragment) getSupportFragmentManager().findFragmentById(R.id.map);
        mapFragment.getMapAsync(new OnMapReadyCallback() {
            @Override
            public void onMapReady(GoogleMap googleMap) {
                Log.d("Map", "지도 준비됨.");
                map = googleMap;

            }
        });

        try {
            MapsInitializer.initialize(this);
        } catch (Exception e) {
            e.printStackTrace();
        }

        final String myLocation = "location";
        final PostResult postResult;
        Button button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                RetrofitService service = retrofit.create(RetrofitService.class);

                try {
                    JSONObject b = new JSONObject();
                    b.put("latitude", latitude);
                    b.put("longitude", longitude);

                    Call<PostResult> call = service.getPosts(new geoData(54.12323, 121.12312));
                    call.enqueue(new Callback<PostResult>() {
                        @Override
                        public void onResponse(Call<PostResult> call, retrofit2.Response<PostResult> response) {
                            if(response.isSuccessful()){
                                Log.d("성공", response.body().toString());
                                nowWeather = response.body().getNowWeather();
                                Toast toast = Toast.makeText(getApplicationContext(), Integer.toString(nowWeather), Toast.LENGTH_LONG);
                                toast.show();
                                //postResult = (postResult) response.getClass();
                            }
                            else{
                                Log.d("실패","값 반환 실패");
                            }

                        }

                        @Override
                        public void onFailure(Call<PostResult> call, Throwable t) {

                        }
                    });
                } catch (JSONException e) {
                    e.printStackTrace();
                }


                //Toast toast = Toast.makeText(getApplicationContext(), Double.toString(latitude), Toast.LENGTH_LONG);
                //toast.show();
                //Toast toast2 = Toast.makeText(getApplicationContext(), Double.toString(longitude), Toast.LENGTH_LONG);
                //toast2.show();

                Intent intent = new Intent(getApplicationContext(), WeatherActivity.class);
                intent.putExtra("local", nowLocal);
                intent.putExtra("nowWeather", nowWeather);
                //intent.putExtra("latitude", latitude);
                //intent.putExtra("longitude", longitude);

                if(val == -1) mes();
                else {startActivity(intent);}

            }
        });

        AutoPermissions.Companion.loadAllPermissions(this, 101);
        //Spinner spinner1 = (Spinner) findViewById(R.id.spinner1);

        final List<String> areas1 = new ArrayList<String>();

        DatabaseReference rootReference = FirebaseDatabase.getInstance().getReference();
        DatabaseReference cityReference = rootReference.child("");


        rootReference.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {

                for (DataSnapshot areaSnapshot : snapshot.getChildren()) {
                    String areaName = areaSnapshot.child("서울특별시").getValue(String.class);
                    areas1.add(areaName);
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                Log.w("TAG: ", "Failed to read value", error.toException());
            }
        });

        AutoPermissions.Companion.loadAllPermissions(this, 101);
        final ArrayAdapter<CharSequence> adspin1 = ArrayAdapter.createFromResource(MainActivity.this, R.array.도시들, android.R.layout.simple_spinner_dropdown_item);
        adspin1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        Spinner spinner1 = (Spinner) findViewById(R.id.spinner1);
        spinner1.setAdapter(adspin1);
        spinner1.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                final String strCity = adspin1.getItem(position).toString();
                //if (position == 0) startLocationService();
                if (adspin1.getItem(position).equals(strCity)) {
                    final ArrayAdapter<CharSequence> adspin2;
                    String packageName = getPackageName();
                    int resId = getResources().getIdentifier(strCity,"array",packageName);

                    adspin2 = ArrayAdapter.createFromResource(MainActivity.this, resId, android.R.layout.simple_spinner_dropdown_item);

                    adspin2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                    Spinner spinner2 = (Spinner) findViewById(R.id.spinner2);
                    spinner2.setAdapter(adspin2);
                    spinner2.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                        @Override
                        public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                            final String strCity2 = adspin2.getItem(position).toString();
                            if (adspin2.getItem(position).equals(strCity2)) {
                                final ArrayAdapter<CharSequence> adspin3;
                                String packageName = getPackageName();
                                int resId2 = getResources().getIdentifier(strCity2,"array",packageName);
                                adspin3 = ArrayAdapter.createFromResource(MainActivity.this, resId2, android.R.layout.simple_spinner_dropdown_item);
                                adspin3.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                                Spinner spinner3 = (Spinner) findViewById(R.id.spinner3);
                                spinner3.setAdapter(adspin3);
                                spinner3.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                                    @Override
                                    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                                        final String strCity3 = adspin3.getItem(position).toString();
                                        if (adspin3.getItem(position).equals(strCity3)) {
                                            final ArrayAdapter<CharSequence> adspin4;
                                            String packageName3 = getPackageName();
                                            int resId3 = getResources().getIdentifier(strCity3,"array",packageName3);
                                            adspin4 = ArrayAdapter.createFromResource(MainActivity.this, resId3, android.R.layout.simple_spinner_dropdown_item);
                                            adspin4.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                                            Spinner spinner4 = (Spinner) findViewById(R.id.spinner4);
                                            spinner4.setAdapter(adspin4);
                                            spinner4.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                                                @Override
                                                public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                                                    final String strCity4 = adspin4.getItem(position).toString();
                                                    if(adspin4.getItem(position).equals(strCity4)){
                                                        final ArrayAdapter<CharSequence> adspin5;
                                                        String packageName3 = getPackageName();
                                                        int resId5 = getResources().getIdentifier(strCity4,"array",packageName3);
                                                        adspin5 = ArrayAdapter.createFromResource(MainActivity.this, resId5, android.R.layout.simple_spinner_dropdown_item);
                                                        latitude = Double.parseDouble(adspin5.getItem(0).toString());
                                                        longitude = Double.parseDouble(adspin5.getItem(1).toString());
                                                    }

                                                    gogoMap(latitude, longitude, strCity+" "+strCity2+" "+strCity3, strCity4);
                                                    nowLocal = strCity+" "+strCity2+" "+strCity3;
                                                    val = 1;
                                                }

                                                @Override
                                                public void onNothingSelected(AdapterView<?> parent) {

                                                }
                                            });

                                        } else if (adspin3.getItem(position).equals("우장산역")) {
                                            gogoMap(37.549293, 126.836634, "서울 강서구", "우장산역");
                                            val = 2;
                                        }
                                    }

                                    @Override
                                    public void onNothingSelected(AdapterView<?> parent) {

                                    }
                                });
                            }
                        }

                        @Override
                        public void onNothingSelected(AdapterView<?> parent) {

                        }
                    });
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });

    }

    public void gogoMap(double latitude, double longitude, String lo, String cation) {

        // 맵 위치를 이동하기
        CameraUpdate update = CameraUpdateFactory.newLatLng(new LatLng(latitude, longitude));
        // 위도,경도
        map.moveCamera(update);

        // 보기 좋게 확대 zoom 하기
        CameraUpdate zoom = CameraUpdateFactory.zoomTo(15);
        map.animateCamera(zoom);

        // 마커 표시 하기 : 위치지정, 풍선말 설정
        MarkerOptions markerOptions = new MarkerOptions()
                // 마커 위치
                .position(new LatLng(latitude, longitude))
                .title(lo) // 말풍선 주 내용
                .snippet(cation); // 말풍선 보조내용


        // 마커를 추가하고 말풍선 표시한 것을 보여주도록 호출
        map.addMarker(markerOptions).showInfoWindow();

    }

    public void startLocationService() {
        LocationManager manager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);

        try {
            Location location = manager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
            if (location != null) {
                double local_latitude = location.getLatitude();
                double local_longitude = location.getLongitude();
                String message = "최근 위치 -> Latitude : " + local_latitude + "\nLongitude:" + local_longitude;

                Log.d("Map", message);
            }

            GPSListener gpsListener = new GPSListener();
            long minTime = 1000000000;
            float minDistance = 0;

            manager.requestLocationUpdates(LocationManager.GPS_PROVIDER, minTime, minDistance, gpsListener);

        } catch (SecurityException e) {
            e.printStackTrace();
        }
    }

    class GPSListener implements LocationListener {
        public void onLocationChanged(Location location) {
            Double local_latitude = location.getLatitude();
            Double local_longitude = location.getLongitude();

            String message = "내 위치 -> Latitude : " + local_latitude + "\nLongitude:" + local_longitude;
            Log.d("Map", message);

            showCurrentLocation(local_latitude, local_longitude);
        }

        public void onProviderDisabled(String provider) {
        }

        public void onProviderEnabled(String provider) {
        }

        public void onStatusChanged(String provider, int status, Bundle extras) {
        }
    }

    private void showCurrentLocation(Double local_latitude, Double local_longitude) {
        LatLng curPoint = new LatLng(local_latitude, local_longitude);
        map.animateCamera(CameraUpdateFactory.newLatLngZoom(curPoint, 15));

        showMyLocationMarker(curPoint);
    }

    private void showMyLocationMarker(LatLng curPoint) {
        if (myLocationMarker == null) {
            myLocationMarker = new MarkerOptions();
            myLocationMarker.position(curPoint);
            myLocationMarker.title("● 내 위치\n");
            myLocationMarker.snippet("● GPS로 확인한 위치");
            //myLocationMarker.icon(BitmapDescriptorFactory.fromResource(R.drawable.mylocation));
            map.addMarker(myLocationMarker);
        } else {
            myLocationMarker.position(curPoint);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        AutoPermissions.Companion.parsePermissions(this, requestCode, permissions, this);
    }

    @Override
    public void onDenied(int requestCode, String[] permissions) {
        //Toast.makeText(this, "permissions denied : " + permissions.length, Toast.LENGTH_LONG).show();
    }

    @Override
    public void onGranted(int requestCode, String[] permissions) {
        //Toast.makeText(this, "permissions granted : " + permissions.length, Toast.LENGTH_LONG).show();
    }

    public void mes() {
        Toast.makeText(this, "위치를 선택하세요", Toast.LENGTH_LONG).show();
    }

}
