from pathlib import Path

config_sections = {
    'prediction'
}

prediction_keys = {
    'demand_input',
    'events_input',
    'data_directory',
    'start_date',
    'stop_date',
    'lags',
    'number_of_grid_rows',
    'number_of_grid_columns',
    'x_min',
    'x_max',
    'y_min',
    'y_max',
    'border_width',
    'number_of_filters',
    'kernel_size',
    'batch_size',
    'learning_rate',
    'step_size',
    'epochs',
    'crs'
}


def sanity_check_config(config: dict):
    for section in config_sections:
        if section not in config:
            raise ValueError(f'{section} is not present in the given configuration. '
                             f'Please add this section to the configuration.')
    for prediction_key in prediction_keys:
        if prediction_key not in config['prediction']:
            raise ValueError(f'{prediction_key} is not present in the given configuration. '
                             f'Please add this parameter to the configuration.')

    if isinstance(config['prediction']['data_directory'], str):
        config['prediction']['data_directory'] = Path(config['prediction']['data_directory']).resolve()


def get_data_start_date(start_hour=0):
    return f'2015-03-09 {start_hour:02d}:00:00'
