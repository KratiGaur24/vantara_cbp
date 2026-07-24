import os
import pandas as pd

def download_and_inspect():
    # Use the absolute path to make sure Windows never loses the file
    file_path = r"D:\Personal\Vantara\vantara_cbp\data\raw\online_retail_II.xlsx"
    
    print(f"📂 Loading Excel sheets from local directory: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ Error: Could not find 'online_retail_II.xlsx' inside data/raw/!")
        print("Please ensure you downloaded it and placed it in the right folder.")
        return None

    # Load both transactional sheets as specified in Section 5.5 of the PRD
    print("⏳ Reading Sheet 1 (Year 2009-2010)... This may take a moment.")
    df_1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
    
    print("⏳ Reading Sheet 2 (Year 2010-2011)...")
    df_2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")
    
    # Merge them chronologically into a single master transaction table
    print("🔗 Concatenating both sheets...")
    df = pd.concat([df_1, df_2], ignore_index=True)
    
    # Sort dates chronologically
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df.sort_values(by='InvoiceDate').reset_index(drop=True)
    
    print("\n✅ Local Data successfully loaded!")
    print(f"📊 Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    download_and_inspect()