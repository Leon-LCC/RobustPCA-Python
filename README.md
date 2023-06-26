# Python Implementation of Robust PCA

## Quick Start
Install numpy and you're good to go! The functions are implemented in [RobustPCA.py](./RobustPCA.py).
```python
# Robust PCA via the Exact ALM Method
Low_Rank_M, Sparse_M = RPCA(Data, Lambda, mu, rho):
# Robust PCA via the Inexact ALM Method
Low_Rank_M, Sparse_M = RPCA_inexact(Data, Lambda, mu, rho):
```


## Application
If you're interested in the details of these applications, please refer to the [report](./Applications.pdf).

- Part 1: [Video Denoising](#part-1-video-denoising)
    - Extract the noise from a noisy video.
    - Separate the foreground (moving objects) and background (stationary objects) from a video. 
- Part 2: [Anomaly Detection](#part-2-anomaly-detection)
    - Use robust PCA to detect anomalies in a dataset of images.


#### Requirements
- numpy
- pandas
- scikit-video
- tqdm


#### Part 1: Video Denoising
- Data
```bash
-- dataset
  |-- Part1
     |-- boat_GT.jpg       'Ground truth of boat.mp4'
     |-- boat.mp4          'Boat video'
     |-- flower_GT.mp4     'Ground truth of flower.mp4'
     |-- flower.mp4        'Flower video'
     |-- TrainStation.mp4  'Train station video'
```
- Usage
```bash
python Part1.py --i input_video_path \
               --o output_video_path \
               --l lambda (optional) \
               --r rho (optional) \
               --save_noise (optional) \
               --all (optional)

E.g., python Part1.py --i ./dataset/Part1/boat.mp4 \
                      --o ./boat_denoised.mp4 \
                      --l 0.1 \
                      --save_noise
```


#### Part 2: Anomaly Detection
- Data 
```bash
-- dataset
  |-- Part2
     |-- Train_data.npy    'MNIST Images'
     |-- Train_label.npy   'Labels (0: normal, 1: anomaly)'
```
- Usage
```bash
python Part2.py --i input_data_path \
                --g input_label_path \
                --o output_csv_path \
                --l lambda (optional) \
                --r rho (optional) \
                --t threshold (optional) \

E.g., python Part2.py --i ./dataset/Part2/train_data.npy \
                      --g ./dataset/Part2/train_label.npy \
                      --o ./detection_result.csv \
                      --l 0.053 \
                      --t 0.999999
```
