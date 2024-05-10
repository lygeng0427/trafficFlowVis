import json

def read_json_file(filepath):
    """
    Reads a JSON file and returns the data as a dictionary.
    
    Args:
    filepath (str): The path to the JSON file to be read.
    
    Returns:
    dict: The dictionary representation of the JSON data.
    """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("The file was not found.")
        return {}
    except json.JSONDecodeError:
        print("The file is not in a proper JSON format.")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


if __name__ == "__main__":
    filepath = "/scratch/lg3490/tfv/sequential_items.json"
    data = read_json_file(filepath)
    print(data)