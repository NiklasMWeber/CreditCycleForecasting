import csv

def importData(string):
    print('Hello '+string)
    
def importFile(fileWithPath, delimiter):
    x = []
    i=0
    with open(fileWithPath, 'r') as file:
        file = csv.reader(file, delimiter=delimiter)
        for row in file:
            x.append(row)
    return x