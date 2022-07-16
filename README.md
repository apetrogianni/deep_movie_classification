# Deep Movie Classification

## Instructions
```shell
virtualenv env
source env/bin/activate
```

## Install requirements

```shell
pip install -r requirements.txt
```

You can download the data from the ["movie_shots_by_experiment"](https://drive.google.com/drive/folders/1saDBlGxu9SxtYkesu5G14W_zvXy1d5Bv?usp=sharing) folder, which contains all the .mp4 files _(along with the .npy files created after the feature_extraction process)_ for the movie shots, divided into 4 different experiments.

<details><summary>Experiments</summary>
<p> 

Experiment | Number of classes
| :--- | ---: 
Binary   | Non_Static (818 shots) <br /> Static (985 shots)
3_class  | Zoom (152 shots) <br />  Static (985 shots) <br /> Vertical_and_horizontal_movements (342 shots)
4_class  | Tilt (89 shots) <br /> Panoramic (253 shots) <br />Static (985 shots) <br /> Zoom (152 shots)
10_class | Static (985 shots) <br /> Panoramic (207 shots) <br /> Zoom in (51 shots) <br /> Travelling_out (46 shots) <br /> Vertical_movements (52 shots) <br /> Aerial (51 shots)<br /> Travelling_in (55 shots)<br /> Tilt (37 shots)<br /> Handled (273 shots)<br /> Panoramic_lateral (46 shots)

</p>
</details>








