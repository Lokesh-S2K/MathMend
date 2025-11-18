import csv
import os

# --- Configuration ---
# The path to your dataset from the root of your project
CSV_FILE_PATH = os.path.join('data', 'dataset.csv') 
EXPECTED_COLUMNS = 2
# ---------------------

print(f"--- Starting CSV Debugger for: {CSV_FILE_PATH} ---")
print(f"Expecting {EXPECTED_COLUMNS} columns per row.\n")

try:
    with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as file:
        # csv.reader gives us row-by-row control
        reader = csv.reader(file)
        
        # Check the rest of the file
        for i, row in enumerate(reader, start=1): # Start from line 1
            # Check if the number of columns is what we expect
            if len(row) != EXPECTED_COLUMNS:
                print(f"--- 🚨 ERROR FOUND 🚨 ---")
                print(f"Mismatch on physical line number: {i}")
                print(f"Expected {EXPECTED_COLUMNS} columns, but the parser found {len(row)}.")
                print("\nHere is the raw content of the problematic row:")
                print(row)
                print("\nSuggestion:")
                print("Open the CSV in a plain text editor (like VS Code or Notepad++), go to this line, and look for extra commas or issues with double-quotes.")
                break # Stop after finding the first error
        else: # This 'else' belongs to the 'for' loop
            print("\n--- ✅ SUCCESS ---")
            print("The CSV file appears to be well-formed. All rows have the correct number of columns.")

except FileNotFoundError:
    print(f"!! FILE ERROR: The file was not found at '{CSV_FILE_PATH}'. Please check the path.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")