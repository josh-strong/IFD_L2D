#!/bin/bash

# URL of the file to download
URL="https://zenodo.org/records/7884735/files/organs_axial.zip?download=1"

# Output filename
OUTPUT_FILE="organs_axial.zip"

# Destination folder for extraction
DEST_FOLDER="organs_axial_data"

# Download the file
echo "Downloading data from $URL..."
wget -O "$OUTPUT_FILE" "$URL"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed: $OUTPUT_FILE"

    # Check if the file is a ZIP archive
    if [[ "$OUTPUT_FILE" == *.zip ]]; then
        echo "Extracting $OUTPUT_FILE to $DEST_FOLDER..."
        
        # Create destination folder if it doesn't exist
        mkdir -p "$DEST_FOLDER"
        
        # Extract the ZIP file
        unzip -q "$OUTPUT_FILE" -d "$DEST_FOLDER"

        rm "$OUTPUT_FILE"
        
        if [ $? -eq 0 ]; then
            echo "Extraction completed: Files are in $DEST_FOLDER"
            
            # Ask user if they want to create expert split
            echo ""
            read -p "Do you want to create expert split for the dataset? (y/n): " create_expert
            if [[ "$create_expert" =~ ^[Yy]$ ]]; then
                echo "Creating expert split..."
                python3 preprocess.py
                if [ $? -eq 0 ]; then
                    echo "Expert split created successfully!"
                else
                    echo "Error occurred while creating expert split"
                fi
            fi
        else
            echo "Error occurred while extracting $OUTPUT_FILE"
        fi
    else
        echo "The downloaded file is not a ZIP archive."
    fi
else
    echo "Download failed. Please check the URL or your network connection."
fi