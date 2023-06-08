import csv
result={}

#Load the CSV file
with open('../LeafScan-back/leafscan/diseases.txt', 'r') as data:
    csvreader = csv.reader(data, delimiter=';')
    d = {rows[0].strip():rows[1].strip() for rows in csvreader}

def disease_info(disease_name):
    '''
    return corresponding data for a given disease
    '''
    result['plant_name'] = disease_name.split("__")[0]
    result['disease_name'] = disease_name.split("__")[1].replace("_", " ")
    result['url']=d[disease_name.strip()]
    return result

print(disease_info('Apple___Cedar_apple_rust'))
