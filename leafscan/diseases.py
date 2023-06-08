import csv
result={}

#Load the CSV file
with open('diseases.csv', delimiter=';' 'r') as data:
  for line in csv.DictReader(data):
      print(line)



def disease_info(disease_name):
    '''
    return corresponding data for a given disease
    '''
    result['plant_name'] = disease_name.split("__")[0]
    result['disease_name'] = disease_name.split("__")[1].replace("_", " ")
    return result

disease_name = 'Apple___Cedar_apple_rust'
disease_info(disease_name)
