# Deep Movie Classification

Two different methods are evaluated; one _static_, which is based on aggregated statistics on the feature sequence, and one _sequential_, that tries to predict the target class based on the input frame sequence. The former adopts an SVM algorithm with the appropriate data normalization and parameter tuning, while for the latter, an LSTM architecture was chosen. In order to obtain features representing the visual characteristics of a movie shot _(for the .mp4 files)_, the "[multimodal_movie_analysis](https://github.com/tyiannak/multimodal_movie_analysis)" repo was used. 

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
> Clone "[multimodal_movie_analysis](https://github.com/tyiannak/multimodal_movie_analysis)" repo for the feature extraction process

### 1.2 Install requirements
```shell
sudo apt install ffmpeg
```
```shell
pip3 install -r requirements.txt
```
### 1.3 Download the data

You can download the data from the ["movie_shots_by_experiment"](https://drive.google.com/drive/folders/1saDBlGxu9SxtYkesu5G14W_zvXy1d5Bv?usp=sharing) folder, which contains all the .mp4 files _(along with the .npy files created after the feature_extraction process)_ for the movie shots, divided into 4 different experiments.

## 2. Train 

By combining different shot categories; four classification tasks, one binary and three multi-label, are defined.

<details><summary>Experiments</summary>
<p> 

Experiment | Classes
| :--- | ---: 
2_class   | Non_Static (818 shots) <br /> Static (985 shots)
3_class  | Zoom (152 shots) <br />  Static (985 shots) <br /> Vertical_and_horizontal_movements (342 shots)
4_class  | Vertical_movements (89 shots) <br /> Panoramic (253 shots) <br />Static (985 shots) <br /> Zoom (152 shots)
10_class | Static (985 shots) <br /> Panoramic (207 shots) <br /> Zoom in (51 shots) <br /> Travelling_out (46 shots) <br /> Vertical_static (52 shots) <br /> Aerial (51 shots)<br /> Travelling_in (55 shots)<br /> Vertical_moving (37 shots)<br /> Handled (273 shots)<br /> Panoramic_lateral (46 shots)

</p>
</details>

## 2.1 Sequential method

i.e. To train the LSTM model, for the 3-class experiment:

```shell
python3 src/train.py -v home/3_class/Zoom home/3_class/Static home/3_class/Vertical_and_horizontal_movements
```

> where _"home/3_class/<class_name>"_ is the full path of the class-folder, containing the .mp4 files

To get aggregated results for a specific number of folds use the flag "-f". For example, for 10-folds:

```shell
python3 src/train.py -v home/3_class/Zoom home/3_class/Static home/3_class/Vertical_and_horizontal_movements -f 10
```

The following files will be saved:
 * `best_checkpoint.pt` the best model
 * `3_class_best_model.pkl` the model parameteres & hyperparameters
<!--  * `LSTM_3_class_y_pred.npy` the posteriors
 * `LSTM_3_class_y_test.npy` the actual values -->











