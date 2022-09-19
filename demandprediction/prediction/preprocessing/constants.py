import datetime


# relevant dates for timechange
time_change_dates = [
    '2015-03-29 02:00:00',
    '2015-10-25 02:00:00',
    '2016-03-27 02:00:00',
    '2016-10-30 02:00:00',
    '2017-03-26 02:00:00',
    '2017-10-29 02:00:00',
    '2018-03-25 02:00:00',
    '2018-10-28 02:00:00',
    '2019-03-31 02:00:00',
    '2019-10-27 02:00:00'
]

# list of tuples containing start and end datum of the octoberfest
octoberfest_dates = [
    (datetime.datetime(2015, 9, 19, 0, 0, 0), datetime.datetime(2015, 10, 4, 23, 59, 59)),
    (datetime.datetime(2016, 9, 17, 0, 0, 0), datetime.datetime(2016, 10, 3, 23, 59, 59)),
    (datetime.datetime(2017, 9, 16, 0, 0, 0), datetime.datetime(2017, 10, 3, 23, 59, 59)),
    (datetime.datetime(2018, 9, 22, 0, 0, 0), datetime.datetime(2018, 10, 7, 23, 59, 59)),
    (datetime.datetime(2019, 9, 21, 0, 0, 0), datetime.datetime(2019, 10, 6, 23, 59, 59))
]

# list of tuples containing start and end datum of the winter holidays
winter_holidays = [
    (datetime.datetime(2015, 2, 14, 0, 0, 0), datetime.datetime(2015, 2, 22, 23, 59, 59)),
    (datetime.datetime(2016, 2, 6, 0, 0, 0), datetime.datetime(2016, 2, 14, 23, 59, 59)),
    (datetime.datetime(2017, 2, 25, 0, 0, 0), datetime.datetime(2017, 3, 5, 23, 59, 59)),
    (datetime.datetime(2018, 2, 10, 0, 0, 0), datetime.datetime(2018, 2, 18, 23, 59, 59)),
    (datetime.datetime(2019, 3, 2, 0, 0, 0), datetime.datetime(2019, 3, 10, 23, 59, 59))
]

# list of tuples containing start and end datum of the easter holidays
easter_holidays = [
    (datetime.datetime(2015, 3, 28, 0, 0, 0), datetime.datetime(2015, 4, 12, 23, 59, 59)),
    (datetime.datetime(2016, 3, 19, 0, 0, 0), datetime.datetime(2016, 4, 3, 23, 59, 59)),
    (datetime.datetime(2017, 4, 8, 0, 0, 0), datetime.datetime(2017, 4, 23, 23, 59, 59)),
    (datetime.datetime(2018, 3, 24, 0, 0, 0), datetime.datetime(2018, 4, 8, 23, 59, 59)),
    (datetime.datetime(2019, 4, 13, 0, 0, 0), datetime.datetime(2019, 4, 28, 23, 59, 59))
]

# list of tuples containing start and end datum of the whitsun holidays
whitsun_holidays = [
    (datetime.datetime(2015, 5, 23, 0, 0, 0), datetime.datetime(2015, 6, 7, 23, 59, 59)),
    (datetime.datetime(2016, 5, 14, 0, 0, 0), datetime.datetime(2016, 5, 29, 23, 59, 59)),
    (datetime.datetime(2017, 6, 3, 0, 0, 0), datetime.datetime(2017, 6, 18, 23, 59, 59)),
    (datetime.datetime(2018, 5, 19, 0, 0, 0), datetime.datetime(2018, 6, 3, 23, 59, 59)),
    (datetime.datetime(2019, 6, 8, 0, 0, 0), datetime.datetime(2019, 6, 23, 23, 59, 59))
]

# list of tuples containing start and end datum of the summer holidays
summer_holidays = [
    (datetime.datetime(2015, 8, 1, 0, 0, 0), datetime.datetime(2015, 9, 14, 23, 59, 59)),
    (datetime.datetime(2016, 7, 30, 0, 0, 0), datetime.datetime(2016, 9, 12, 23, 59, 59)),
    (datetime.datetime(2017, 7, 29, 0, 0, 0), datetime.datetime(2017, 9, 11, 23, 59, 59)),
    (datetime.datetime(2018, 7, 28, 0, 0, 0), datetime.datetime(2018, 9, 10, 23, 59, 59)),
    (datetime.datetime(2019, 7, 27, 0, 0, 0), datetime.datetime(2019, 9, 9, 23, 59, 59))
]

# list of tuples containing start and end datum of the autumn holidays
autumn_holidays = [
    (datetime.datetime(2015, 10, 31, 0, 0, 0), datetime.datetime(2015, 11, 8, 23, 59, 59)),
    (datetime.datetime(2016, 10, 29, 0, 0, 0), datetime.datetime(2016, 11, 6, 23, 59, 59)),
    (datetime.datetime(2017, 10, 28, 0, 0, 0), datetime.datetime(2017, 11, 5, 23, 59, 59)),
    (datetime.datetime(2018, 10, 27, 0, 0, 0), datetime.datetime(2018, 11, 4, 23, 59, 59)),
    (datetime.datetime(2019, 10, 26, 0, 0, 0), datetime.datetime(2019, 11, 3, 23, 59, 59))
]

# list of tuples containing start and end datum of the christmas holidays
christmas_holidays = [
    (datetime.datetime(2015, 12, 24, 0, 0, 0), datetime.datetime(2016, 1, 7, 23, 59, 59)),
    (datetime.datetime(2016, 12, 24, 0, 0, 0), datetime.datetime(2017, 1, 5, 23, 59, 59)),
    (datetime.datetime(2017, 12, 23, 0, 0, 0), datetime.datetime(2018, 1, 7, 23, 59, 59)),
    (datetime.datetime(2018, 12, 22, 0, 0, 0), datetime.datetime(2019, 1, 6, 23, 59, 59)),
    (datetime.datetime(2019, 12, 21, 0, 0, 0), datetime.datetime(2020, 1, 6, 23, 59, 59))
]

# dict containing the previous holiday lists
holidays_dict = {
    'winter': winter_holidays,
    'easter': easter_holidays,
    'whitsun': whitsun_holidays,
    'summer': summer_holidays,
    'autumn': autumn_holidays,
    'christmas': christmas_holidays
}
