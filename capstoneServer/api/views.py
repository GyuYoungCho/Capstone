import json
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import permissions
from rest_framework.parsers import JSONParser
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
import time
from bson import json_util, ObjectId

@api_view(['GET'])
def ping(request):
    if request.method == 'GET':
        return JsonResponse({
            "message" : "ok"
        })

@api_view(['POST'])
def geography(request):
    if request.method == 'POST':
        jsonVal = request.data
        latitude = jsonVal['latitude']
        longitude = jsonVal['longitude']
        nowWeather = 0  #현재 날씨
        nowroad = 0     #현재 도로 상황
        rain = 0
        ### return "weather" - tag : nowWeather, rain , "videoUrl" - tag : videoUrl, "road" - tag: not yet
        #test#
        '''
        #v_url = getVideoUrl(latitude, longitude)
        #img = bring_Image(latitude, longitude)
        #weather, rains = pred(img)
        #video = "http://cctvsec.ktict.co.kr/138/q3qr5/LAGa2yRFrcntssdB4uydBh6uHM4ZuSeIiGIC+fuLak2r0QGAMmtHdRSmGi"
        
        return JsonResponse({
            "latitude" : latitude,
            "longitude" : longitude,
            "nowWeather" : nowWeather,
            "nowroad" : nowroad,
            "rain" : rain,
            "videoUrl" : video
        })
        '''
        ##actual
        
        weather = 1000
        rain = 1000
        v_url = getVideoUrl(latitude, longitude)
        img = bring_Image(latitude, longitude)
        weather, rain = pred(img)
        nowroad = bring_speed(latitude, longitude)
        print(nowroad)
        
        return JsonResponse({
            "latitude" : latitude,
            "longitude" : longitude,
            "nowWeather" : weather,
            "nowroad" : nowroad,
            "rain" : rain,
            "videoUrl" : v_url
        })
        
        
###
from rest_framework.views import APIView
#from .apps import WebappConfig
from keras.preprocessing import image
import numpy as np
from tensorflow import keras

#
#import tensorflow as tf

def pred(img):
    #with tf.device('/CPU:0'):
    img = np.expand_dims(img,axis=0)
    img = img.astype('float32')        
    img /= 255.
    predictor = keras.models.load_model('./api/model_test/best_sig_db_l3_weather_ce_vgg19.h5')
    #pred = WebappConfig.predictor.predict(img)  
    hat = predictor.predict(img)
    del predictor
    weather = 0
    rain = 0.0

    if hat[0][0] > hat[0][1] :
        if hat[0][0] > hat[0][2] :
            weather = 0
        else :
            weather = 3
    else :
        if hat[0][1] > hat[0][2]:
            weather = 1
        else :
            weather = 3

    if weather == 3:
        predictor = keras.models.load_model('./api/model_test/best_sig_db_l3_rain_mse_vgg16.h5') 
        rain = predictor.predict(img)[0][0]
        del predictor

    if weather == 1:
        predictor = keras.models.load_model('./api/model_test/best_sig_db_l3_fog_ce_vgg19.h5') 
        hat = predictor.predict(img)
        if hat[0][0] > hat[0][1] :
            if hat[0][0] > hat[0][2] :
                weather = 0
            else :
                weather = 2
        else :
            if hat[0][1] > hat[0][2]:
                weather = 2
            else :
                weather = 1
        del predictor

    return weather,rain

class call_model(APIView):
    def post(self,request):
        if request.method == 'POST':
            #local test
            image_path = "C:/Users/user/Desktop/CAP/source/models/data/image96/맑음/2020102517_[경부선] 공세육교_frame0.jpg"
            img = image.img_to_array(image.load_img(image_path,target_size=(96,96)))
            img = np.expand_dims(img,axis=0)
            img /= 255.
            #img = img.astype('float32')
            #print(img)
            predictor = keras.models.load_model('./api/model_test/best_sig_db_l3_weather_ce_vgg19.h5')
            #pred = WebappConfig.predictor.predict(img)  
            hat = predictor.predict(img)
            del predictor
            weather = 0
            rain = 0.0
            
            if hat[0][0] > hat[0][1] :
                if hat[0][0] > hat[0][2] :
                    weather = 0
                else :
                    weather = 3
            else :
                if hat[0][1] > hat[0][2]:
                    weather = 1
                else :
                    weather = 3
            
            if weather == 3:
                predictor = keras.models.load_model('./api/model_test/best_sig_db_l3_rain_mse_vgg16.h5') 
                rain = predictor.predict(img)[0][0]
                del predictor
            
            if weather == 1:
                predictor = keras.models.load_model('./api/model_test/best_sig_db_l3_fog_ce_vgg19.h5') 
                hat = predictor.predict(img)
                if hat[0][0] > hat[0][1] :
                    if hat[0][0] > hat[0][2] :
                        weather = 0
                    else :
                        weather = 2
                else :
                    if hat[0][1] > hat[0][2]:
                        weather = 2
                    else :
                        weather = 1
                del predictor
             
            return JsonResponse({'nowWeather' : weather, 'rain' : rain})
            
            
from .api import download
from bs4 import BeautifulSoup
import requests
import io
from PIL import Image

headers = { 'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"}

api_key = '1602398376506'

def getVideoUrl(latitude,longitude):
    #jsonVal = request.data
    #latitude = jsonVal['latitude']
    #logitude = jsonVal['longitude']
    cctv_video_url = 'http://openapi.its.go.kr:8081/api/NCCTVInfo'
    params = {
        'key': api_key,
        'ReqType': '2',
        'type ': 'its',
        'MinX': longitude - 0.0001,
        'MaxX': longitude + 0.0001,
        'MinY': latitude - 0.0001,
        'MaxY': latitude + 0.0001,
    }
    resp = download(cctv_video_url, params, headers, 'POST')
    dom = BeautifulSoup(resp.text, 'html.parser')
    url = dom.find('data').find('cctvurl').text.strip()
    print(url)
    #return JsonResponse({'videoUrl' : url})
    return url


def bring_Image(latitude, longitude):
    #jsonVal = request.data
    #latitude = jsonVal['latitude']
    #logitude = jsonVal['longitude']
    cctv_video_url = 'http://openapi.its.go.kr:8081/api/NCCTVInfo'
    api_key = '1602398376506'
    cctv_image_url = 'http://openapi.its.go.kr:8081/api/NCCTVImage'
    params = {
        'key': api_key,
        'ReqType': '2',
        'type ': 'its',
        'cctvtype' : '3',
        'MinX': longitude - 0.0001,
        'MaxX': longitude + 0.0001,
        'MinY': latitude - 0.0001,
        'MaxY': latitude + 0.0001,
    }
    resp = download(cctv_image_url, params, headers, 'POST')
    dom = BeautifulSoup(resp.text, 'html.parser')
    url = dom.find('data').find('cctvurl').text.strip()

    data = requests.get(url).content
    img = Image.open(io.BytesIO(data))
    img = img.resize((96,96), Image.BILINEAR)
    img = np.array(img)
    #print(img)
    #return JsonResponse({'image': img})
    return img

def bring_speed(latitude, longitude):
    #if request.method == 'POST':
    #jsonVal = request.data
    #latitude = jsonVal['latitude']
    #longitude = jsonVal['longitude']
    traffic_url = 'http://openapi.its.go.kr/api/NTrafficInfo'
    
    key = '1602002870218'
    params = {
        'key': key,
        'ReqType': '2',
        'MinX': longitude - 0.005,
        'MaxX': longitude + 0.005,
        'MinY': latitude - 0.005,
        'MaxY': latitude + 0.005,
    }
    resp = download(traffic_url, params, headers, 'POST')
    dom = BeautifulSoup(resp.text, 'html.parser')

    speed = []
    for i in dom.find_all('data'):
        speed.append(int(i.find('avgspeed').text.strip()))

    return int(np.mean(speed))
