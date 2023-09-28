from src.module_1.module_1_meteo_api import (
    get_dict_values_for_year,
    calculate_var,
    calculate_mean,
    calculate_std,
)


def test_get_dict_values_for_year():
    days_list = ["2023-12-29", "2023-12-30", "2023-12-31", "2024-12-30", "2024-12-31"]
    data_list = [1, 2, 3, 4, 5]

    expected_res = {"2023": [1, 2, 3], "2024": [4, 5]}

    assert get_dict_values_for_year(days_list, data_list) == expected_res


def test_calculate_var():
    dict_values_for_year = {"2023": [1, 2], "2024": [2, 4]}

    expected_res = {"2023": 0.25, "2024": 1}

    assert calculate_var(dict_values_for_year) == expected_res


def test_calculate_mean():
    dict_values_for_year = {"2023": [1, 2], "2024": [2, 4]}

    expected_res = {"2023": 1.5, "2024": 3}

    assert calculate_mean(dict_values_for_year) == expected_res


def test_calculate_std():
    dict_values_for_year = {"2023": [1, 2], "2024": [2, 4]}

    expected_res = {"2023": 0.5, "2024": 1}

    assert calculate_std(dict_values_for_year) == expected_res
