import pandas as pd
import requests

class DataReader:

    def __init__(self, api_url, estimated_timezone):
        self.api_url = "https://api.meteo.lt/v1/places/"
        self.estimated_timezone = estimated_timezone  # Replace with actual zone if known

    def get_historical_data(self, from_date, to_date):

        params = {
            "from_date": from_date,
            "to_date": to_date
        }
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()  # Assuming JSON response

            # Parse data into DataFrame (adapt based on your API's response structure)
            df = pd.DataFrame(data["data"])  # Assuming data key holds actual data
            df.index = pd.to_datetime(df.index)  # Assuming timestamps are in the index
            df.index = df.index.tz_localize(self.estimated_timezone)  # Set estimated time zone

            return df
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None