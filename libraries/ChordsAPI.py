# import dependencies
import urllib.request
import pandas as pd
import os

class ChordsAPI:
    def __init__(self, domain):
        self.domain         = domain
        self.base_api_dir   = 'api/v1/data/'
        self.local_data_dir = 'csv_files'

        # make sure that the local data directory exists
        os.makedirs(self.local_data_dir, exist_ok=True)
        


    def download_csv_file(self, instrument_id, start, end):
        # define the URL for the CHORDS API
        url = f'http://{self.domain}/{self.base_api_dir}/{instrument_id}.csv?start={start}&end={end}'
        
        print(url)

        with urllib.request.urlopen(url) as f:
            data = f.read().decode('utf-8')

        return(data)


    # Create the standard file name pattern for the specified instrument and dates
    def get_file_name(self, instrument_id, start_str, end_str):
        readable_domain = self.domain.replace(".", "_")

        file_name = f'{self.local_data_dir}/{readable_domain}_instrument_id_{instrument_id}_{start_str}_to_{end_str}.csv'
        return(file_name)


    # Download all the data for the specified instrument and dates
    # This is done one day at a time in order to not overwhelm the CHORDS portal
    # The individual files are then concatenated in to one big file
    def get_csv_data(self, instrument_id, start_str, end_str):
        
        # Define a list of each individual day from start to end
        start_dates = pd.date_range(start=start_str, end=end_str)
        end_dates = start_dates + pd.DateOffset(days=1)

        file_name = self.get_file_name(instrument_id, start_str, end_str)

        f = open(file_name, 'w')
        
        print("Downloading data for instrument id ", instrument_id, " for dates from ", start_str, " to ", end_str)


        # for each individual day, download the csv for that one day
        for index, start_date in enumerate(start_dates):
            end_date = end_dates[index]

            # FORMAT: 2021-01-03T00:00
            start = start_date.strftime("%Y-%m-%dT00:00")
            end = end_date.strftime("%Y-%m-%dT00:00")

            # Download the csv file
            data_str = self.download_csv_file(instrument_id, start, end)

            # parse the data into individual lines
            lines = data_str.splitlines()

            # Write the header if this is the first downloaded file
            if index == 0:
              header = "\n".join(lines[0:19])
              f.write(header + "\n")

            # write the rest of the data (skipping the header)
            f.write("\n".join(lines[20:len(lines)]) + "\n")

            
        f.close
        
        print("Download complete, file created: ", file_name)