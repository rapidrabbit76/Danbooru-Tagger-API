# Danbooru Tagger API

[Danbooru tagger](https://github.com/danbooru/danbooru) api using FastAPI & service-streamer


# Requirements

- torch==1.10.1
- torchvision==0.11.2
- fastapi[all]==0.80.0
- fastapi-restful==0.4.3
- fastapi-health==0.4.0
- service-streamer==0.1.2
- pydantic==1.9.2



# API 

### /predict/embedding

**Inputs** : 
  - image file 

**Outputs** : 
  - float list (embedding)


### /predict/score

**Inputs** : 
  - image file 

**Outputs** : 
  - float list (tag score 0~1)


### /predict/score

**Inputs** : 
  - image file 
  - threshold (score threshold)

**Outputs** : 
  - float list (6000, tag score 0~1)



# Environment Value

```bash
# env setting is in 
>> ./app/settings.py
```

| Name                     | Default                      | Desc                                              |
| ------------------------ | ---------------------------- | ------------------------------------------------- |
| DL_EMBEDDING_MODEL_PATH  | "model_store/embedding.zip"  | tagger embedding model part                       |
| DL_CLASSIFIER_MODEL_PATH | "model_store/classifier.zip" | tagger classifier model part                      |
| MB_BATCH_SIZE            | 64                           | Micro Batch: MAX Batch size                       |
| MB_MAX_LATENCY           | 0.2                          | MAX Latency between API calls                     |
| MB_WORKER_NUM            | 1                            | Process count                                     |
| CUDA_DEVICE              | "cpu"                        | target cuda device                                |
| CUDA_DEVICES             | [0]                          | visible cuda device                               |
| CORS_ALLOW_ORIGINS       | [*]                          | cross origin resource sharing setting for FastAPI |


# RUN from code

## 1. install req.txt
```bash
>> cd /REPO/ROOT/DIR/PATH
>> pip install -r requirements.txt
```

## 2. download models and split 
```bash
>> TODO: 
```

## 3. run API by uvicorn
```bash
>> cd /REPO/ROOT/DIR/PATH
>> uvicorn app.main:app --host 0.0.0.0
```


# RUN using Docker

## 1. Image Build
```bash
>> cd /REPO/ROOT/DIR/PATH
>> docker-compose build
```

## 1. Image pull
```bash
>> docker pull ys1lee/danbooru-tag-api:latest
```

## 2. Container RUN
```bash
>> cd /REPO/ROOT/DIR/PATH
>> docker-compose up -d 
```


# curl cmd

![](./src/temp.jpeg)

```bash
>> curl -X 'POST' \
  'http://localhost:8000/predict/tag' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@temp.jpeg' \
  -F 'threshold=0.2'

["otohime_(youngest_princess)","tani_takeshi","gogiga_gagagigo","ha_akabouzu","shimazaki_mujirushi","pageratta","wakabayashi_toshiya","tonda","mochi_au_lait","eroe","shichimenchou","bai_lao_shu","public_nudity","public","shiranui_mai","rappa_(rappaya)","ryouou_school_uniform","hoshizuki_(seigetsu)","kyonko","nishi_koutarou","pool_ladder","abuse","gym_storeroom","denpa_onna_to_seishun_otoko","jin_(mugenjin)","shirou_masamune","showering","haramura_nodoka","exhibitionism","niiko_(gonnzou)","gaoo_(frpjx283)","abubu","yui_(angel_beats!)","deepthroat","dennou_coil","as109","ikamusume","eromanga","takara_miyuki","kimi_kiss","thai_text","miyanaga_saki","freediving","tsuda_nanafushi","goma_(gomasamune)","minami_(colorful_palette)","ino","swim_trunks","kusanagi_motoko","sonya_(kill_me_baby)","kasumi_(doa)","poolside","lm_(legoman)]
```


