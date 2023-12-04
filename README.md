# TTS HiFiGAN project
### Aksenov Yaroslav

## Installation guide

Download [model](https://disk.yandex.ru/d/28qe10g33LdGzQ) to the folder ```final_model```

Create folder ```"data/LJSpeech-1.1"``` with LJSpeech dataset
```shell
pip install -r ./requirements.txt
```

## Launching guide

#### Testing:
   ```shell
   python test.py \
      -c src/train_config.json \
      -o generated \
      -r final_model/hifigan_ckpt.pth \
      -wp wavs
   ```

#### Training:
   ```shell
   python train.py \
      -c src/train_config.json \
      -wk "YOUR_WANDB_API_KEY"
   ```

#### Test outputs
Audios for test melspecs are available in [generated](generated) directory