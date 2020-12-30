package org.techtown.capstone;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Headers;
import retrofit2.http.POST;

public interface RetrofitService {
    @Headers("Content-Type: application/json")
    @POST("api/v1/geo/")
    Call<PostResult> getPosts(@Body geoData body);
}