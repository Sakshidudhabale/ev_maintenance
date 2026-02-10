import zipfile

pkl_file = "ev_model.pkl"          
zip_file = "ev_model.zip"         

with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(pkl_file)

print("✅ PKL file zipped successfully!")
