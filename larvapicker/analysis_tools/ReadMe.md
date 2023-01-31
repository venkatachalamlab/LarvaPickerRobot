## About

The Larva Picker is an automated system for handling *Drosophila* larvae for continuous long-term observation.
Follow this ReadMe for instructions on how to analyze the data obtained with the robot.

## Table of Contents
1. [Tracking larvae](#parser)
2. [Tracking posture](#posture_tracker)
3. [Classifying behavior](#state_tracker)
4. [Compiling behavioral features](#compiler)

---

<a name="parser"></a>
## Tracking larvae 

**1. Check saved files**

Check that your dataset has been saved with this structure:

```
└── dataset_path
│   ├── data.h5
│   ├── log.json
│   ├── metadata.json
│   └── movie.mp4
```


**2. RUN `parser`**

Use the following command:
```bash
parser --dataset_path=<dataset_path> --n_larva=<n_larva>
```

You can use the movie or log saved with the data to manually check the number of larvae present. There should be at least one frame where the correct number of larvae are visible at once.

The parser will track each individual larva throughout the video and generate a new data file with isolated crops around each larva.

The resulting file structure should look like this:

```
├── dataset_path
│   ├── data.h5
│   ├── log.json
│   ├── metadata.json
│   └── movie.mp4
├── 0
│   ├── data.h5
│   ├── coordinates.h5
│   └── movie.mp4
├── 1
│   ├── data.h5
│   ├── coordinates.h5
│   └── movie.mp4
...
```

---

<a name="posture_tracker"></a>
## Tracking posture 

**3. RUN `posture_tracker`**

Use the following command:
```bash
posture_tracker --dataset_path=<dataset_path> --checkpoint_path=<checkpoint_path> --batch_size=<batch_size>
```

`checkpoint_path` should point to a PyTorch file `checkpoint.pt` that contain the trained model weights for the PostureTracker neural network.

`batch_size` can vary depending on the hardware's memory limits, but ~200 is a good default.

The PostureTracker will analyze each crop to identify the head, tail, and midspine of the larva. 
This information is saved in `pt_results.h5` and will be passed into the StateTracker. 
You can always check the results by reviwing the generated movies, `pt_pred.mp4` (raw probabilities from network) and `pt_results.mp4` (final coordinates for posture points).

---

<a name="state_tracker"></a>
## Classifying behavior

**4. RUN `state_tracker`**

Use the following command:
```bash
state_tracker --dataset_path=<dataset_path> --checkpoint_path=<checkpoint_path> --batch_size=<batch_size>
```

`checkpoint_path` should point to a PyTorch file `checkpoint.pt` that contain the trained model weights for the StateTracker neural network.

`batch_size` can vary depending on the hardware's memory limits, but ~5000 is a good default.

The StateTracker will analyze the coordinates produced by the parser and PostureTracker to classify each frame into a behavioral state ("RUN" or "TURN"). 
This information is saved in `st_results.h5` and wil be passed into the compiler. 
You can always check the results by reviwing the generated trajectory marked with the corresponding behavior state, `st_results.png`.


---

<a name="compiler"></a>
## Compiling behavioral features

**5. Create an index of datasets to be compiled together.**

> This step is **OPTIONAL** if you are only analyzing a single dataset.

Create a file named `index.json` and populate it as a jsonified dictionary of paths to datasets that will be compiled together into behavioral statistics and features.

**6. RUN `compiler`**

Use the following command:
```bash
compiler (--dataset_path=<dataset_path> | --index_path=<index_path>)
```
> Note that you cannot specify both arguments. 
> Choose `dataset_path` to analyze a single dataset, or specify the path to your `index.json` in `index_path` to compile multiple datasets together.

The compiler will take all results (`coordinates.h5`, `pt_results.h5` and `st_results.h5`) to calculate behavioral quantities for each larva, and compile all the resulting data together to calculate behavioral statistics for the experiment. 
The results for individual larvae will be saved in their respective directories.
The compiled results will be saved in `larvapicker/analysis_tools/compiler/bin`.

