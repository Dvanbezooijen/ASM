import pandas as pd
import xml.etree.ElementTree as ET

def load_cpt_data(file_path):
    # Load the XML file
    xml_file = file_path
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Namespace dictionary to handle XML namespaces
    namespaces = {
        "cptcommon": "http://www.broservices.nl/xsd/cptcommon/1.1",
        "swe": "http://www.opengis.net/swe/2.0"
    }
    
    # Locate the <cptcommon:values> tag containing the data
    values_element = root.find(".//cptcommon:values", namespaces)
    if values_element is not None:
        data_text = values_element.text.strip()
        
        # Extract the encoding details to determine separators
        encoding_element = root.find(".//swe:encoding", namespaces)
        if encoding_element is not None:
            text_encoding = encoding_element.find("swe:TextEncoding", namespaces)
            if text_encoding is not None:
                decimal_separator = text_encoding.attrib.get("decimalSeparator", ".")
                token_separator = text_encoding.attrib.get("tokenSeparator", ",")
                block_separator = text_encoding.attrib.get("blockSeparator", ";")
        
        # Split the data into rows and columns
        rows = data_text.split(block_separator)
        data = [row.split(token_separator) for row in rows if row]
        
        # Create a DataFrame with all columns
        df = pd.DataFrame(data)
        
        # Define column indices based on known structure
        column_indices = [0, 1, 2, 3, 18]  # Selecting only meaningful columns
        column_names = [
            "Depth (m)", "Depth (m) Duplicate", "Cone Resistance (qc, kPa)", 
            "Sleeve Friction (fs, kPa)", "Friction Ratio (%)"
        ]
        
        # Select only relevant columns
        df = df.iloc[:, column_indices]
        df.columns = column_names
        
        # Attempt to convert columns to numeric types
        df = df.apply(pd.to_numeric, errors='coerce')
        
        #print(df.head())  # Display first few rows
    else:
        print("No CPT data found in the XML file.")
    return df
        