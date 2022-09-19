import json
import logging
import configparser
from pathlib import Path
from contextlib import suppress

import click
import pandas as pd
import pyfiglet
from termcolor import colored

import demandprediction.utils.shared_filenames as filenames
from demandprediction.utils.lazy_loader import LazyLoader
from demandprediction.utils.utils import sanity_check_config

model_package = LazyLoader(
    local_name='model_package',
    parent_module_globals=globals(),
    name='demandprediction.prediction.model'
)

model_selection = {
    '0': 'ConvLSTM',
    '1': 'ConvLSTM with Conv2D for Events',
    '2': 'SARIMAX(2,0,1)(2,0,2,4)(n)',
    '3': 'SARIMAX(2,0,1)(0,0,0,0)(None)',
    '4': 'BorderModel',
    '5': 'Citywide Model',
    '6': 'Border division SARIMAX(2,0,1)(0,0,0,0)(None)',
    '7': 'Border division SARIMAX(2,0,1)(2,0,2,3)(n)',
    '8': 'Border division SARIMAX(2,0,1)(2,0,2,36)(n)',
    '9': 'Full LSTM Border Model',
    '10': 'Full multi-step LSTM Border Model',
    '11': 'ConvLSTM Multistep',
    '12': 'HistoricalAverage',

}

DEFAULT_CONFIG = {
    'prediction': {
        'demand_input': './data/example/historic_demand_munich_example.csv',
        'events_input': './data/example/events.csv',
        'data_directory': str(Path().absolute().joinpath('data')),
        'start_date': '2018-12-31 00:00:00',
        'stop_date': '2020-01-06 06:00:00',
        'lags': 36,
        'number_of_grid_rows': 8,
        'number_of_grid_columns': 8,
        'x_min': 687412,
        'x_max': 695412,
        'y_min': 5331314,
        'y_max': 5339314,
        'number_of_filters': 30,
        'kernel_size': 2,
        'batch_size': 24,
        'learning_rate': 0.001,
        'step_size': 20,
        'epochs': 500,
        'prediction_horizon': 1
    }
}


def setup_logging():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(name)-30s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_demand(config: dict, overwrite=False):
    """
    Load the demand from a local file.
    """
    save_path = config['data_directory'].joinpath(
        filenames.get_trips_filename(
            stop_date=config['stop_date']
        )
    )
    if overwrite or not save_path.is_file():
        df = pd.read_csv(config['demand_input'])
        # Filter out unwanted timestamps
        df = df[(df.timestamp_start > config['start_date']) & (df.timestamp_start <= config['stop_date'])]
        df.to_hdf(save_path, key='df', complevel=9)


def load_events(config: dict, overwrite=False):
    """
    Loads the events from a local file.
    """
    save_path = config['data_directory'].joinpath(
        filenames.get_events_filename(
            stop_date=config['stop_date']
        )
    )
    if overwrite or not save_path.is_file():
        df = pd.read_csv(config['events_input'])
        # Filter out unwanted timestamps
        df = df[(df.timestamp_start > config['start_date']) & (df.timestamp_start <= config['stop_date'])]
        df.to_hdf(save_path, key='df', complevel=9)


def model_lifecycle(config: dict, model_id: int, max_workers=None):
    # Create the model based on the selected model_id
    if model_id == 0:
        model = model_package.ConvLSTMModel(config)
    elif model_id == 1:
        model = model_package.ConvLSTMEventsConv2DModel(config)
    elif model_id == 2:
        model = model_package.SARIMAXModel(config, (2, 0, 1), (2, 0, 2, 4), 'n', max_workers)
    elif model_id == 3:
        model = model_package.SARIMAXModel(config, (2, 0, 1), (0, 0, 0, 0), None, max_workers)
    elif model_id == 4:
        model = model_package.BorderModel(config)
    elif model_id == 5:
        model = model_package.CitywideModel(config)
    elif model_id == 6:
        model = model_package.SARIMAXBorderModel(config, (2, 0, 1), (0, 0, 0, 0), None, max_workers)
    elif model_id == 7:
        model = model_package.SARIMAXBorderModel(config, (2, 0, 1), (2, 0, 2, 3), 'n', max_workers)
    elif model_id == 8:
        model = model_package.SARIMAXBorderModel(config, (2, 0, 1), (2, 0, 2, 36), 'n', max_workers)
    elif model_id == 9:
        model = model_package.LSTMBorderModel(config)
    elif model_id == 10:
        model = model_package.LSTMBorderModelMultiStep(config)
    elif model_id == 11:
        model = model_package.BorderModelMultistep(config)
    elif model_id == 12:
        model = model_package.HistoricalAverage(config)
    elif model_id == 13:
        model = model_package.PersistenceModel(config)
    else:
        raise ValueError("A non valid model was chosen!")
    # Execute the model lifecycle
    model.prepare_data()
    model.generate_model()
    model.train_model(plot_fitting_history=True)
    model.predict()
    model.save_predictions()
    model.score_model()


def main(config: dict, model_id: int, max_workers=None):
    welcome_message = colored(pyfiglet.figlet_format('Demand-\nprediction', font='slant'), color='green')
    print(welcome_message)

    # Initialize prerequisites
    setup_logging()
    sanity_check_config(config)
    prediction_config = config['prediction']
    filenames.setup_paths(prediction_config['data_directory'])

    # Load needed data
    load_demand(config=prediction_config)
    load_events(config=prediction_config)

    # Create model and execute the model lifecycle
    model_lifecycle(config=prediction_config, model_id=model_id, max_workers=max_workers)


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option(
    '--config', 'config_path',
    help='Path to configuration file (optional). '
         'All options specified in this file will overwrite the command line options'
)
@click.option(
    '--demand-input', '-d', 'demand_input',
    default=str(DEFAULT_CONFIG['prediction']['demand_input']),
    show_default=True,
    help='A CSV file containing the historic demand.'
)
@click.option(
    '--events-input', '-d', 'events_input',
    default=str(DEFAULT_CONFIG['prediction']['events_input']),
    show_default=True,
    help='A CSV file containing the historic event information.'
)
@click.option(
    '--data-dir', '-d', 'data_directory',
    default=str(DEFAULT_CONFIG['prediction']['data_directory']),
    show_default=True,
    help='The directory where all files will be stored.'
)
@click.option(
    '--start-date',
    default=DEFAULT_CONFIG['prediction']['start_date'],
    show_default=True,
    help='The datetime for the first prediction interval.'
)
@click.option(
    '--stop-date',
    default=DEFAULT_CONFIG['prediction']['stop_date'],
    show_default=True,
    help='The datetime after which no predictions will be made.'
)
@click.option(
    '--grid-rows', 'number_of_grid_rows',
    default=DEFAULT_CONFIG['prediction']['number_of_grid_rows'],
    show_default=True,
    help='The number of rows in the grid.'
)
@click.option(
    '--grid-columns', 'number_of_grid_columns',
    default=DEFAULT_CONFIG['prediction']['number_of_grid_columns'],
    show_default=True,
    help='The number of columns in the grid.'
)
@click.option(
    '--x-min',
    default=DEFAULT_CONFIG['prediction']['x_min'],
    show_default=True,
    help='The lower boundary of the grid in x direction.'
)
@click.option(
    '--x-max',
    default=DEFAULT_CONFIG['prediction']['x_max'],
    show_default=True,
    help='The upper boundary of the grid in x direction.'
)
@click.option(
    '--y-min',
    default=DEFAULT_CONFIG['prediction']['y_min'],
    show_default=True,
    help='The lower boundary of the grid in y direction.'
)
@click.option(
    '--y-max',
    default=DEFAULT_CONFIG['prediction']['y_max'],
    show_default=True,
    help='The upper boundary of the grid in y direction.'
)
@click.option(
    '--lags',
    default=DEFAULT_CONFIG['prediction']['lags'],
    show_default=True,
    help='The number of lags the LSTM part of the model uses.'
)
@click.option(
    '--step-size',
    default=DEFAULT_CONFIG['prediction']['step_size'],
    show_default=True,
    help='The number of minutes between two timeseries timesteps.'
)
@click.option(
    '--filters', 'number_of_filters',
    default=DEFAULT_CONFIG['prediction']['number_of_filters'],
    show_default=True,
    help='The number of filters in the first layer of the model.'
)
@click.option(
    '--kernel-size',
    default=DEFAULT_CONFIG['prediction']['kernel_size'],
    show_default=True,
    help='The size of the kernel in the first layer of the model.'
)
@click.option(
    '--batch-size',
    default=DEFAULT_CONFIG['prediction']['batch_size'],
    show_default=True,
    help='The batch size which is used for training.'
)
@click.option(
    '--lr', 'learning_rate',
    default=DEFAULT_CONFIG['prediction']['learning_rate'],
    show_default=True,
    help='The learning rate which is used for training.'
)
@click.option(
    '--epochs',
    default=DEFAULT_CONFIG['prediction']['epochs'],
    show_default=True,
    help='The maximum number of epochs during training.'
)
@click.option(
    '--prediction_horizon',
    default=DEFAULT_CONFIG['prediction']['prediction_horizon'],
    show_default=True,
    help='The number of prediction timesteps (Only considered for citywide model) '
)
def cli(**kwargs):
    """This script predicts the taxi demand for the last year in munich."""
    parsed_prediction_config = {}
    parsed_database_config = DEFAULT_CONFIG['database'].copy()
    # Parse configuration
    if kwargs['config_path'] is not None:
        parsed_config = configparser.ConfigParser()
        parsed_config.read(kwargs['config_path'])
        if 'prediction' in parsed_config.sections():
            parsed_prediction_config = dict(parsed_config['prediction'])
            for key, value in parsed_prediction_config.items():
                # Try to parse strings to int or float
                with suppress(ValueError):
                    int_value = int(value)
                    parsed_prediction_config[key] = int_value
                    continue
                # noinspection PyUnreachableCode
                with suppress(ValueError):
                    float_value = float(value)
                    parsed_prediction_config[key] = float_value
    kwargs.pop('config_path')

    username = kwargs.pop('username')
    password = kwargs.pop('password')

    kwargs.update(parsed_prediction_config)

    # Ask for the desired model
    selection_string = ''
    for idx, model in model_selection.items():
        selection_string = selection_string + idx + ": " + model + '\n'
    selected_model = click.prompt(f'Which model do you want to use?\n{selection_string}', prompt_suffix='\n',
                                  type=click.Choice([i for i in model_selection]))

    max_workers = None
    if 'SARIMAX' in model_selection[selected_model]:
        max_workers = click.prompt('How many CPU cores do you want to use?', type=int)

    click.echo(json.dumps(kwargs, indent=4))
    click.echo(f'Model: {model_selection[selected_model]}')

    # Confirm configuration before starting demandprediction
    click.confirm('Do you want to start the prediction with the shown configuration?',
                  default=True, abort=True)

    # If not given ask for database username and password
    if 'username' not in parsed_database_config:
        if username is not None:
            parsed_database_config['username'] = username
        else:
            parsed_database_config['username'] = click.prompt(
                'Please enter the username for accessing the database'
            )
    if 'password' not in parsed_database_config:
        if password is not None:
            parsed_database_config['password'] = password
        else:
            parsed_database_config['password'] = click.prompt(
                'Please enter the password for accessing the database',
                hide_input=True
            )

    config = {
        'database': parsed_database_config,
        'prediction': kwargs
    }
    # Start main logic
    main(config=config, model_id=int(selected_model), max_workers=max_workers)

    click.echo(click.style('Demand prediction finished successfully!', fg='green'))


if __name__ == '__main__':
    parsed_config = configparser.ConfigParser()
    parsed_config.read('myconfig.ini')
    if 'prediction' in parsed_config.sections():
        parsed_prediction_config = dict(parsed_config['prediction'])
        for key, value in parsed_prediction_config.items():
            with suppress(ValueError):
                int_value = int(value)
                parsed_prediction_config[key] = int_value
                continue
            # noinspection PyUnreachableCode
            with suppress(ValueError):
                float_value = float(value)
                parsed_prediction_config[key] = float_value
    parsed_config._sections['prediction'].update(parsed_prediction_config)
    main(config=parsed_config._sections, model_id=13, max_workers=4)
