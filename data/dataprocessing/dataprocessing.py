import json
import pandas as pd
import os
import urllib



def json_to_txt(jsonfilepath):

    with open(jsonfilepath, 'r') as jsonfile:
        data_df = json.load(jsonfile)

        data_extract = []
        data_extract_txt = ''
        for k in data_df:
            data_extract.append(k['text'])
            data_extract_txt+= k['text']
            

        

        with open('data1.txt', 'w') as file:
            file.write(str(data_extract_txt))




def parquet_to_txt():
    
    # Lire le fichier Parquet
    df = pd.read_parquet('/Users/dossierantoine/Desktop/dl/transformer/ia génération de texte/gpt project/data/datainstruct/train-00000-of-00001-844fbb78a655163e.parquet')

    # Sélectionner la colonne spécifique
    nom_colonne = 'text'  # Remplacez par le nom de votre colonne
    colonne = df[nom_colonne]

    # Écrire le contenu de chaque cellule de la colonne dans un fichier texte
    with open('instructtxt.txt', 'w') as f:
        for valeur in colonne:
            f.write(f"{valeur}\n")

    print(f"Le contenu de chaque cellule de la colonne '{nom_colonne}' a été écrit dans sortie_colonne_complete.txt")




def get_batches_parquet():
    

    # Step 1: Read the Parquet file
    file_path = 'train-00000-of-00001-494661a0baeb14c0.parquet'
    df = pd.read_parquet(file_path)

    # Step 2: Limit the DataFrame to the first 10,000 rows
    df = df.head(20000)

    # Debug: Print the columns in the DataFrame to ensure the correct column name is used
    print("Columns in the DataFrame:", df.columns)

    # Step 3: Extract the desired column
    column_name = 'text'  # Replace with the name of your column
    if column_name in df.columns:
        column_data = df[column_name]
    else:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Debug: Print the first few entries of the column data
    print("First few entries of the column data:")
    print(column_data.head())

    # Step 4: Write to a text file
    output_file_path = 'wikisample.txt'
    with open(output_file_path, 'w') as file:
        for item in column_data:
            # Ensure item is converted to string to avoid any issues
            file.write(f"{str(item)}")

    print(f"Column '{column_name}' from the first 2000 rows has been written to {output_file_path}")

   



def truncate_txt():
    length = 20000000000

    with open('histoiresfr.txt', "r+") as file:
        content = file.readlines()
        compteur = 0
        new_file = ""
        for k in content:
            compteur += len(k)
            if compteur < length:
                new_file+= k
            else:
                break
        with open('histoiresfrecourtees.txt', 'w') as file2:
            file2.write(new_file)





def parquet_to_json(filepath):
    # Read the Parquet file
    df = pd.read_parquet(filepath)

    # Convert the DataFrame to JSON
    json_data = df.to_json(orient='records', indent=4)


    

    # Save the JSON data to a file
    with open('train_0_out_of_7_instructions.json', 'w') as json_file:
        json_file.write(json_data)

    print("The Parquet file has been converted to JSON and saved as output.json")



def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data




def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text




def format_json_file(filepath):
    df = pd.read_json(filepath, orient='records')
    df_dict = df.to_dict(orient='records')
    with open("instruct.txt", 'w') as file:
        for k in df_dict:
            file.write(format_input(k))





def read_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data



def jsonl_to_json(jsonl_file_path):

        # Liste pour stocker les objets JSON
    data = []

    # Lire le fichier JSONL
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            # Convertir chaque ligne en un objet Python et l'ajouter à la liste
            data.append(json.loads(line))

    # Chemin vers le fichier JSON de sortie
    json_file_path = 'wikipedia.json'

    # Écrire la liste des objets dans un fichier JSON
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def truncate_text(filepath, id_truncation):
    with open(filepath, 'r') as file:
        data = file.read()
    dataliste = data.split()
    dataextractslist = [] 
    for k in range(0, len(dataliste), id_truncation):
        dataextractslist.append(dataliste[k:(k+id_truncation)])

    datatowrite = ""

    for k in dataextractslist:
        intermediatestring = ""
        for j in k:
            intermediatestring+=j+" "
        datatowrite+= intermediatestring+"\n"

    with open(f"{filepath}2.txt", 'w') as file2:
        file2.write(datatowrite)
    #return dataextractslist


def writesample(filepath, id_truncation):
    for num in range(len(truncate_text(filepath, id_truncation))):
        with open(f'train{num}.txt', "w") as file:
            txt = ''
            for j in truncate_text(filepath, id_truncation)[num]:
                txt+=(j+' ')
            file.write(txt) 


def melt_text_files(filepath1, filepath2):
    with open(filepath1, 'r') as file1:
        data1 = file1.read()

    with open(filepath2, 'r') as file2:
        data2 = file2.read()

    with open('data.txt', 'w') as file3:
        file3.write((data1+data2))


def json_selection(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    data2 = []
    for k in data:
        dic = {"instruction":k['prompt'], 'output':k['125M']}
        data2.append(dic)
    
    with open(file_path, "w") as file2:
        json.dump(data2, file2, indent=4)


def json_truncation(file_path, id_truncation):
    with open(file_path, 'r') as file:
        data = json.load(file)

    data2 = []
    for k in data:
        
        dic = {"text":k['text']}
        data2.append(dic)
    data3= data2[:id_truncation]
    
    with open(file_path, "w") as file2:
        json.dump(data3, file2, indent=4)



def jsoncleaning(file_path, file_path2):
    with open(file_path, 'r') as file:
        data = json.load(file)

    data2 = []
    for k in data:
        if k["text"] != "":
            if k["text"][1] != '=':
                substring = "@"
                k["text"].replace(substring, "")
                dic = {"text":k['text']}
                data2.append(dic)

    
    with open(file_path2, "w") as file2:
        json.dump(data2, file2, indent=4)

parquet_to_json('/Users/dossierantoine/Desktop/dl/transformer/ia génération de texte/gpt project/data/datainstruct/train_0_out_of_7_instructions.parquet')
