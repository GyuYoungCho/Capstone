package org.techtown.capstone;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class Fragment1 extends Fragment {

    ImageView imageView;

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState){
        ViewGroup rootView = (ViewGroup) inflater.inflate(R.layout.fragment1,container,false);
        imageView = rootView.findViewById(R.id.imageView6);
        return rootView;
        //return inflater.inflate(R.layout.fragment1,container,false);
    }

    public void setImage(int resId){
        imageView.setImageResource(resId);
    }
}