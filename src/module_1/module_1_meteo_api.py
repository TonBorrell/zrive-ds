import requests
from jsonschema import validate

from validate_json_meteo_api import meteo_api_schema

API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"


def validate_json_meteo_api(instance: dict[str, any]):
    validate(instance=instance, schema=meteo_api_schema)


def get_data_meteo_api(city: str):
    city_latitude = COORDINATES[city]["latitude"]
    city_longitude = COORDINATES[city]["longitude"]

    r = requests.get(
        f"{API_URL}latitude={city_latitude}&longitude={city_longitude}&start_date=1950-01-01&end_date=2050-12-31&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily={VARIABLES}"
    )

    if r.status_code == 200:
        validate_json_meteo_api(instance=r.json())


def main():
    print(get_data_meteo_api("Madrid"))


if __name__ == "__main__":
    main()
