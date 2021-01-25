import argparse
import pandas as pd
from datetime import datetime

if __name__=="__main__":

    l=['Chatting','Discussion','GroupStudy','NULL','Presentation']

    for i, item in enumerate(l):
        sample_data=pd.read_csv("{}40.csv".format(item), header=None)
        sample_data.columns=['name','value','timestamp']