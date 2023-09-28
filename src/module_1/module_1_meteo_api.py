import requests
from jsonschema import validate
import numpy as np
import pandas as pd

from schemas.validate_json_meteo_api import meteo_api_schema

API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
INITIAL_DATE = "1950-01-01"
FINAL_DATE = "2050-12-31"
MODELS = "CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S"  # noqa


def get_request(url: str):
    r = requests.get(url)

    # If r checks if answer is not an 4xx or 5xx
    if r:
        return r.json()


def validate_json_meteo_api(instance: dict[str, any]):
    validate(instance=instance, schema=meteo_api_schema)


def get_data_meteo_api(city: str):
    city_latitude = COORDINATES[city]["latitude"]
    city_longitude = COORDINATES[city]["longitude"]

    params = {
        "latitude": city_latitude,
        "longitude": city_longitude,
        "start_date": INITIAL_DATE,
        "end_date": FINAL_DATE,
        "models": MODELS,
        "daily": VARIABLES,
    }

    json_response = get_request(
        f"{API_URL}latitude={city_latitude}&longitude={city_longitude}&start_date={INITIAL_DATE}&end_date={FINAL_DATE}&models={MODELS}&daily={VARIABLES}"  # noqa
    )

    if not json_response:
        raise Exception("Wrong API call")

    validate_json_meteo_api(instance=json_response)

    return json_response


def get_dict_values_for_year(days_list: list[str], data_list: list[float]):
    # Creates a list with all the values for each year
    year_list = {}
    actual_year = []
    for index, i in enumerate(days_list):
        day = i.split("-")[2]
        month = i.split("-")[1]
        year = i.split("-")[0]
        actual_year.append(data_list[index])
        if day == "31" and month == "12":
            year_list[year] = actual_year
            actual_year = []

    return year_list


def calculate_var(dict_values_for_year: dict[str, list[float]]):
    return {year: np.var(dict_values_for_year[year]) for year in dict_values_for_year}


def calculate_std(dict_values_for_year: dict[str, list[float]]):
    return {year: np.std(dict_values_for_year[year]) for year in dict_values_for_year}


def calculate_mean(dict_values_for_year: dict[str, list[float]]):
    return {year: np.mean(dict_values_for_year[year]) for year in dict_values_for_year}


def calculate_data(json: dict[str, any]):
    initial_dict_for_df = {
        "Year": list(
            range(int(INITIAL_DATE.split("-")[0]), int(FINAL_DATE.split("-")[0]) + 1)
        )
    }
    df_mean = pd.DataFrame(initial_dict_for_df)
    df_var = pd.DataFrame(initial_dict_for_df)
    df_std = pd.DataFrame(initial_dict_for_df)
    for key_dict in json["daily"]:
        if key_dict != "time":
            if None not in json["daily"][key_dict]:
                dict_values_for_year = get_dict_values_for_year(
                    json["daily"]["time"],
                    json["daily"][key_dict],
                )
                mean_for_year = calculate_mean(dict_values_for_year)
                df_mean[key_dict] = mean_for_year.values()
                var_for_year = calculate_var(dict_values_for_year)
                df_var[key_dict] = var_for_year.values()
                std_for_year = calculate_std(dict_values_for_year)
                df_std[key_dict] = std_for_year.values()

    return (df_mean, df_var, df_std)


def create_plot_and_save(df, filename, x="Year"):
    plot = df.plot(x=x, figsize=(20, 8)).get_figure()
    plot.savefig(filename)


def main():
    for city in COORDINATES:
        json = get_data_meteo_api(city)
        df_mean, df_var, df_std = calculate_data(json)
        create_plot_and_save(df_mean, f"src/module_1/images/mean_{city}.png")
        create_plot_and_save(df_var, f"src/module_1/images/var_{city}.png")
        create_plot_and_save(df_std, f"src/module_1/images/std_{city}.png")


if __name__ == "__main__":
    main()
