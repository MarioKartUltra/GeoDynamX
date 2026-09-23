"""Data I/O utilities for creepmeter time series."""

import numpy as np


def read_usgs_10min_file(filename):
    import pandas as pd
    columns = ['Year', 'Julian_Day', 'Displacement', 'Quality']
    dtype = {
        'Year': str,
        'Julian_Day': float,
        'Displacement': float,
        'Quality': str
    }
    data = pd.read_csv(
    filename,
    sep=r"\s+",
    engine="python",   # safest for regex separator
    header=None,
    names=columns,
    dtype=dtype
)
    data['Displacement'] = pd.to_numeric(data['Displacement'], errors='coerce')
    return data


def create_signal(data):
    years = data['Year'].astype(int).values
    julian_days = data['Julian_Day'].values - 1  # Convert to 0-based indexing
    amplitude = data['Displacement'].values

    # Convert to fractional years
    time = years + (julian_days / (365 + (years % 4 == 0).astype(int)))
    return time, amplitude


def select_date_range(data, start_date, end_date):
    """
    Select data within a specific date range
    start_date, end_date: strings in format 'YYYY.DDD' where DDD is julian day
    """
    # Parse start and end dates
    start_year, start_julian = map(float, start_date.split('.'))
    end_year, end_julian = map(float, end_date.split('.'))

    # Create boolean mask for date range
    data_years = data['Year'].astype(float)
    data_julian = data['Julian_Day'].values

    mask = ((data_years > start_year) |
            ((data_years == start_year) & (data_julian >= start_julian))) & \
            ((data_years < end_year) |
            ((data_years == end_year) & (data_julian <= end_julian)))

    return data[mask].reset_index(drop=True)
