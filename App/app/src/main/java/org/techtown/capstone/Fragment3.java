package org.techtown.capstone;

import android.graphics.Paint;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.VideoView;

import androidx.fragment.app.Fragment;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

public class Fragment3 extends Fragment {
    VideoView videoView;

    @Override
    public void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View v = inflater.inflate(R.layout.fragment3, container, false);
        videoView= (VideoView) v.findViewById(R.id.videoView);
        TextView textView = v.findViewById(R.id.textView4);
        textView.setPaintFlags(textView.getPaintFlags() | Paint.FAKE_BOLD_TEXT_FLAG);
        final SwipeRefreshLayout swipeRefreshLayout = (SwipeRefreshLayout)v.findViewById(R.id.swipe);
       swipeRefreshLayout.setOnRefreshListener(new SwipeRefreshLayout.OnRefreshListener() {
           @Override
           public void onRefresh() {
               String str = "http://cctvsec.ktict.co.kr/138/q3qr5/LAGa2yRFrcntssdB4uydBh6uHM4ZuSeIiGIC+fuLak2r0QGAMmtHdRSmGi";
               Uri videoUri = Uri.parse(str);
               //videoView.setMediaController(new MediaController());
               videoView.setVideoURI(videoUri);

               swipeRefreshLayout.setRefreshing(false);
           }
       });

        videoView.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mp) {
                videoView.start();
            }
        });

        return v;
    }
}