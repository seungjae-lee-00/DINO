import pandas as pd
import os

svkpi_dir = "/data/udb/GODTest_SVKPI_3000km/JPEGImages"
file_list = [fname for fname in os.listdir(svkpi_dir) if os.path.isfile(os.path.join(svkpi_dir, fname))]
             
df = pd.DataFrame()
df["img"] = file_list
new_df = df.sample(100)
df.to_csv('data_csv/svkpi3000km.csv')
new_df.to_csv('data_csv/svkpi3000km_mini.csv')
