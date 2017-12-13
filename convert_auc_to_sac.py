import csv
import numpy as np
tag_name = np.load('./CS446-project/tag_name.npy')
with open('./auc_sub.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    rowCount = 0
    for row in readCSV:
        retVal = ""
        if rowCount != 0:
            for i in range(len(row)):
                if i==0:
                    retVal += row[i]+","
                elif row[i] == '1':
                    retVal += tag_name[i-1]+" "
            retVal = retVal[:len(retVal)-1]
            print(retVal)
        rowCount+=1
