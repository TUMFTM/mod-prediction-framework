# Mobility on demand prediction-framework

Prediction Model Framework used in the thesis "Predictive Fleet Strategy for Ridesourcing Services in Reference to Munich's Taxi System" (2022) by Michael Wittmann, M. Sc. at the Institute of Automotive Technology (FTM) at Technical University Munich (TUM).

## Setup Project
This project uses poetry for managing the dependencies. You can find the installation instructions for your platform
[here](https://python-poetry.org/docs/#installation).

After cloning the repository, you just need to run `poetry install`, in order to install all required packages.
If you plan to work on the code/repository, please also run `pre-commit install` to install some pre-commit hooks,
so that the repository stays in a clean state.

## Running a prediction model
After running `poetry install`, you can either use the built-in `demandprediction` cli, which runs a prediction model based on the given config file.
If you want to use a config.ini file use `demandpredciton --config myconfig.ini`

If you prefer to run a python script you can also use the `main.py` script.

If you want to use the GPU to train your TensorFlow models you need to set the environment variable targeting your CUDA install directory e.g. `LD_LIBRARY_PATH=:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-10.2/targets/x86_64-linux/lib`.

## Folder and file structure
```bash
demandprediction
├───data 
│   ├───evaluation # generated
│   ├───logs # generated
│   ├───models # generated
│   ├───prediction_analysis # generated
│   ├───predictions # output directory
│   ├───train_test # cache for preprocessed input data
│   └───visualization # generated
├───demandprediction # all important files are in this directory
│   ├───prediction
│   │   ├───generators  # generators for feeding the models with the needed data
│   │   ├───layers  # custom layers for neural networks
│   │   ├───model  # models for generating predictions
│   │   ├───preprocessing  # preprocessing and modification of the base data
│   │   └───postprocessing  # error measures and postprocessing
│   ├───utils  # shared filenames and utilty functions
│   └───main.py  # main script which starts the prediction
├───myconfig.ini  # Configuration File. Models and input data sources can be configured by the parameters in this file
└───pyproject.toml  # Toml file which specifies all project dependencies
```

## Configuration
### Create your own config
Copy the `default_config.ini` and set your parameters:
```bash
cp ./default_contig.ini ./myconfig.ini
```
### Set your parameters
```ini
[prediction]
demand_input=./data/example/historic_demand_munich_example.csv ; A csv file containing the demand information
events_input=./data/example/events.csv ; A csv file containing the event information
crs='EPSG:25832'                ; coordinate system used for input and outputs (Reccomended to use a metric system)
data_directory=./data           ; data directory relative to wkd
start_date=2018-12-31 00:00:00  ; start of prediction interval
stop_date=2020-01-06 06:00:00   ; end of prediction interval
lags=36                         ; number of lags used for prediction. Caution!: lagsize and step_size are dependent
number_of_grid_rows=8           ; number of grid rows
number_of_grid_columns=8        ; number of grid columns
x_min=687412                    ; x_min grid_boarders in EPSG:25832
x_max=695412                    ; x_max grid_boarders in EPSG:25832
y_min=5331314                   ; y_min grid_boarders in EPSG:25832
y_max=5339314                   ; y_max grid_boarders in EPSG:25832
border_width=5000               ; boarder with in m
number_of_filters=30            ; number of filters used in LSTM-Model
kernel_size=2                   ; kernel size of convolution layer
batch_size=24                   ; batch size
learning_rate=0.001             ; learning rate for neuronal networks
step_size=20                    ; time series frequency in min. Caution lagsize and ste_szie are dependent
epochs=500                      ; number of training epochs
prediction_horizon=1            ; number of lags in prediction horizon

```

## Example Dataset
You can find an artificial randomized dateset in `data/example/historic_demand_munich.csv` and `/data/example/events.csv`
> Please Note: This Dataset is for demonstration purposes only and won't produce any meaningful results. 
> It should rather serve as an example how your input data needs to be formatted. 
> The only model that is worth to run with this dataset is the persistence model. 

## Models
### `base_model.py`
Abstract model framework for further implementations
### `border_model.py`
ConvLSTM with n x m square inner grid and 4 border LSTM cells
### `border_model_multistep.py`
ConvLSTM with n x m square inner grid and 4 border LSTM cells with multistep prediction
### `citywide_model.py` 
Global LTSM Model
### `full_lstm_border_model.py`
LSTM with n x m square inner grid and 4 border LSTM cells
### `full_lstm_border_model_multi_step.py` 
LSTM with n x m square inner grid and 4 border LSTM cells with multistep prediction
### `conv_lstm_events_conv2d.py`
ConvLSTM Model with additional event MLP Model
###`conv_lstm_model.py`
n x m square grid lstm model
### `ha_model.py` 
Historical average model
### `persistence_model.py` 
Persistence model 
### `sarimax.py`
SARIMAX Model with n x m square grid
### `sarimax_border_model.py`
SARIMAX Model with n x m square inner grid and 4 border cells


## Contributors
- [Michael Wittmann](https://github.com/michaelwittmann)
- [Maximilian Speicher](https://github.com/maxispeicher)
- Jan Koch


## Citation
If you find our work useful in your research, please consider citing:
```Latex
@book{wittmann2022,
    title = {Prädiktive Flottenstrategie für Ridesourcing-Dienste am Beispiel des Münchner Taxiverkehrs},
    author = {Wittmann, Michael},
    year = {2022},
    isbn = {##########}
}
```
