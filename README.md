# Deep Movie Classification

In this work we present an original, publicly available [dataset](https://github.com/magcil/movie_shot_classification_dataset.git) for film shot type classification that is  associated with the distinction across 10 types of camera movements that cover the vast majority of types of shots in real movies. We propose two distinct classification methods that can give an intuition about the separability of these categories. 

Two different methods are evaluated; one _static_, which is based on aggregated statistics on the feature sequence, and one _sequential_, that tries to predict the target class based on the input frame sequence. The former adopts an SVM algorithm with the appropriate data normalization and parameter tuning, while for the latter, an LSTM architecture was chosen. In order to obtain features representing the visual characteristics of a movie shot _(for the .mp4 files)_, the [multimodal_movie_analysis](https://github.com/tyiannak/multimodal_movie_analysis) repo was used. 

## 1 Setup

### 1.1 Add "multimodal movie analysis" as a submodule
<!--
```shell
virtualenv env
source env/bin/activate
```
> Use a virtual environment-->

<!-- ```shell
git submodule add https://github.com/tyiannak/multimodal_movie_analysis.git multimodal_movie_analysis
``` -->
```shell
git submodule init 
git submodule update
```
> Clones the "[multimodal_movie_analysis](https://github.com/tyiannak/multimodal_movie_analysis)" repo for the feature extraction process

### 1.2 Install requirements
```shell
sudo apt install ffmpeg
```
```shell
pip3 install -r requirements.txt
```
<!-- ### 1.3 The data

You can download the data from the [movie_shots_by_experiment](https://drive.google.com/drive/folders/1saDBlGxu9SxtYkesu5G14W_zvXy1d5Bv?usp=sharing) folder, which contains all the .mp4 files _(along with the .npy files created after the feature extraction process)_ for the movie shots, divided into 4 different experiments _(check "Experiments" below)_. -->

## 2. Train 

By combining different shot categories; four _(one binary and three multi-label)_ classification tasks are defined.

<details><summary>Tasks</summary>
<p> 

Task | Classes
| :--- | ---: 
2_class   | Non_Static (818 shots) <br /> Static (985 shots)
3_class  | Zoom (152 shots) <br />  Static (985 shots) <br /> Vertical_and_horizontal_movements (342 shots)
  4_class  | Tilt (89 shots) <br /> Panoramic (253 shots) <br />Static (985 shots) <br /> Zoom (152 shots)
10_class | Static (985 shots) <br /> Panoramic (207 shots) <br /> Zoom in (51 shots) <br /> Travelling_out (46 shots) <br /> Vertical_static (52 shots) <br /> Aerial (51 shots)<br /> Travelling_in (55 shots)<br /> Vertical_moving (37 shots)<br /> Handheld (273 shots)<br /> Panoramic_lateral (46 shots)

</p>
</details>

The experiments were conducted using the [dataset](https://github.com/magcil/movie_shot_classification_dataset.git) mentioned above after creating 4 different tasks:

* The ```2_class``` task includes the **Static** and **Non-static** classes. The former consists of shots that have been annotated as static, while the latter contains all the classes from the original dataset that are associated with any type of camera movement. That is the corresponding sub-classes are: _Panoramic Lateral, Vertical Static, Zoom-in, Handheld, Aerial, Vertical Moving, Panoramic, Travelling-in, Travelling-out_. 

* The ```3_class``` task includes the _Static_, _Zoom_ and _Vertical & Horizontal Movements_. The **Static** class is the one that was used in the aforementioned binary task. The **Zoom** class consists of the _Zoom-in, Travelling-in and Travelling-out_ sub-classes, which all contain shots in which the perimeter image changes at very fast intervals, while the centre image remains static or changes at a slower rate. The **Vertical \& Horizontal Movements** class consists of the _Vertical Static, Vertical Moving, Panoramic and Panoramic Lateral_ sub-classes from the original dataset, where the position of the camera is moving either in a vertical or in a horizontal way.

* The ```4_class``` task includes the **Static** and **Zoom** classes of the 3-class problem, while the Vertical \& Horizontal Movements class was separated into 2 sub-classes; **Tilt**, which includes all vertical movements and consists of the _Vertical Static_ and _Vertical Moving_ original classes, and **Panoramic** that contains shots with lateral movements and consists of the _Panoramic_ and _Panoramic Lateral_ original classes.

* The ```10-class``` task includes all provided classes from the original dataset; **Static, Panoramic, Zoom-in, Travelling-out, Vertical Static,
Aerial, Travelling-in, Vertical Moving, Handheld and Panoramic Lateral**.


### 2.1 Sequential method

The LSTM model was trained for the 4 classification tasks that were mentioned above using the [sequential_features](https://github.com/magcil/movie_shot_classification_dataset/tree/main/sequential_features).

i.e. For the 3-class task:


```shell
cd src
python3 train.py -v home/3_class/Zoom home/3_class/Static home/3_class/Vertical_and_horizontal_movements
```

> where _"home/3_class/<class_name>"_ is the full path of the class-folder, containing the .mp4 files

To get aggregated results for a specific number of folds use the flag "-f". For example, for 10-folds:

```shell
python3 train.py -v home/3_class/Zoom home/3_class/Static home/3_class/Vertical_and_horizontal_movements -f 10
```

The following files will be saved:
 * `best_checkpoint.pt` the best model
 * `3_class_best_model.pkl` the model's parameteres & hyperparameters
<!--  * `LSTM_3_class_y_pred.npy` the posteriors
 * `LSTM_3_class_y_test.npy` the actual values -->

## 3. Inference

Four pretrained models are saved in the [pretrained_models](https://github.com/apetrogianni/deep_movie_classification/tree/main/pretrained_models) folder. Each one can be loaded for inference. While in ```/src``` folder:

```shell
python3 inference.py -i <input> -m <../pretrained_models/2_class_best_checkpoint.pt>
```
>,where \<input> is the full path of the .mp4 file or a folder of .mp4 files that you want to classify, and <../pretrained_models/2_class_best_checkpoint.pt> is the path of the pretrained model you want to use for the prediction.







