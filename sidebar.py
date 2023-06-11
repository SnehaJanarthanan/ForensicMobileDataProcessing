import face_recognition
import os
import shutil
import streamlit as st
from PIL import Image
import csv
from textblob import TextBlob
import re
import boto3
import pandas as pd
import base64
import easyocr
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
import easyocr
import streamlit as st
import base64
import numpy as np


def perform_ocr_and_generate_text_file():
    # Check if the output text file exists and delete it if it does
    output_file_path = "output_text.txt"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # Specify the path to the folder containing the images
    camera_roll = r"C:\Users\APQA\Desktop\Hi\tech\vitall\suspect_images"

    # Initialize the EasyOCR reader
    # You can specify the languages you want to support
    reader = easyocr.Reader(['en'])

    # Get the current directory
    current_dir = os.getcwd()

    # Create the 'imageocr' directory if it doesn't exist
    output_dir = os.path.join(current_dir, "imageocr")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the output text file within the 'imageocr' directory
    output_file_path = os.path.join(output_dir, output_file_path)

    # Iterate through all the images in the folder
    with open(output_file_path, "w") as output_file:
        for filename in os.listdir(camera_roll):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_path = os.path.join(camera_roll, filename)

                # Perform OCR on the image and extract the text
                result = reader.readtext(image_path)
                if result:
                    # Write the image name and extracted text to the output file
                    output_file.write(filename + "\n")
                    for entry in result:
                        output_file.write(entry[1] + " ")
                    output_file.write("\n")

    return output_file_path


def main6():
    st.title("Image OCR")

    st.text("Click the button to perform OCR on images and generate a text file.")

    if st.button("Perform OCR"):
        output_file_path = perform_ocr_and_generate_text_file()

        st.success("OCR completed! Text file generated.")

        # Provide a download link for the text file
        with open(output_file_path, 'r') as file:
            text_data = file.read()
            b64_text_data = base64.b64encode(
                text_data.encode()).decode('utf-8')
            href = f'<a href="data:text/plain;base64,{b64_text_data}" download="output_text.txt">Download Text File</a>'
            st.markdown(href, unsafe_allow_html=True)


def extract_verification_codes(text):
    pattern = r"\b\d{6}\b"  # Matches 6-digit verification codes
    codes = re.findall(pattern, text)
    return codes


def detect_and_display_violent_images():
    # Read access keys from CSV file
    with open(r'C:\Users\APQA\Desktop\Hi\tech\vitall\vitall\passworddetails.csv') as file:
        next(file)
        access_key_id, secret_access_key = file.readline().strip().split(',')

    # Create AWS Rekognition client
    client = boto3.client('rekognition', region_name='ap-south-1',
                          aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

    # Folder containing images
    folder_path = r"C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\images"

    # Output folder for saving violent images
    output_folder = r"C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\violentimages"

    # Check if the output folder already exists and delete it
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List to store image paths of violent images
    violent_image_paths = []

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        # Add more extensions if needed
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)

            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()

            # Detect violent content in the image
            response = client.detect_moderation_labels(
                Image={'Bytes': image_bytes}, MinConfidence=50)

            moderation_labels = response['ModerationLabels']
            if moderation_labels:
                for label in moderation_labels:
                    if label['Name'] == 'Violence':
                        # Save the image in the output folder
                        output_image_path = os.path.join(
                            output_folder, filename)
                        shutil.copy(image_path, output_image_path)
                        violent_image_paths.append(output_image_path)
                        break

    return violent_image_paths


def main5():
    st.title("Violent Image Detection")
    st.text("Click the button to detect and display violent images from the camera roll")

    if st.button("Detect Violent Images"):
        violent_images = detect_and_display_violent_images()

        if violent_images:
            st.success("Image analysis complete! Violent images found.")

            # Display the output images
            st.text("Violent Images:")
            for image_path in violent_images:
                st.image(image_path, use_column_width=True)

            # Create a ZIP file of the output images
            output_zip_path = shutil.make_archive(
                "violent_images", 'zip', r"C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\violentimages")

            # Provide a download link for the ZIP file
            with open(output_zip_path, 'rb') as file:
                zip_data = file.read()
                b64_zip_data = base64.b64encode(zip_data).decode('utf-8')
                href = f'<a href="data:application/zip;base64,{b64_zip_data}" download="violent_images.zip">Download Violent Images</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("No violent images found.")


def extract_website(text):
    pattern = r"from\s+(\S+)"  # Matches "from" followed by the website
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""


def filter_otp_messages(df):
    df['OTPs'] = df['Body'].apply(lambda x: extract_verification_codes(str(x)))
    df = df[df['OTPs'].apply(lambda x: len(x) > 0 and len(x[0]) == 6)]
    return df[['OTPs', 'Address']]


def process_csv_bank_messages(file_path, selected_bank):
    df = pd.read_csv(file_path)
    bank_messages = df[df['Address'].str.contains(
        selected_bank, case=False)]
    selected_columns = ['Address', 'updateAt', 'Body']
    bank_messages_selected = bank_messages[selected_columns]
    return bank_messages_selected


def process_csv_search_address(file_path, address):
    df = pd.read_csv(file_path)
    filtered_data = df[df['Address'].str.contains(address)]
    selected_columns = ['Address', 'updateAt', 'Body']
    return filtered_data[selected_columns]


def main4():
    file_path = r'C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\sms\sms_log.csv'

    st.title("Search messages by Address")
    address = st.text_input("Enter Address")

    if st.button("Fetch Messages"):
        if address:
            messages = process_csv_search_address(file_path, address)
            st.dataframe(messages)
        else:
            st.warning("Please enter an address.")

    st.title("Bank Message Extractor")
    banks = ['VK-KOTAKB', 'VM-IndBnk', 'SBIINB', 'ATMSBI']
    selected_bank = st.selectbox("Select Bank", banks)

    if st.button("Fetch Bank Messages"):
        bank_messages = process_csv_bank_messages(file_path, selected_bank)
        st.dataframe(bank_messages)

    st.title("OTPs and Senders")

    if st.button("Fetch OTPs"):
        df = pd.read_csv(file_path)
        filtered_data = filter_otp_messages(df)
        st.dataframe(filtered_data)


def main3():

    st.header("Hate speech recognition")
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv('SMS-Data.csv')

    # Define a regular expression pattern for hate speech
    hate_speech_pattern = r"(?i)\b(hate|kill|otp|attack|discriminate|racist)\b"

    # Function to check if a text contains hate speech
    def check_hate_speech(text):
        if isinstance(text, str):
            match = re.search(hate_speech_pattern, text)
            if match:
                return True
        return False

    # Get user input for the text to search
    search = st.text_input("Enter the text you want to search for")

    # Create a button
    button_clicked = st.button("Search")

    if button_clicked:
        # Drop rows with missing values (NaN) in the 'text' column
        data = data.dropna(subset=['text'])

        # Apply the hate speech detection function to the 'text' column
        data['hate_speech'] = data['text'].apply(check_hate_speech)

        # Filter the DataFrame based on the search text
        if search:
            filtered_data = data[data['text'].str.contains(
                search, case=False, na=False)]
        else:
            filtered_data = data

        # Calculate the percentage of hate speech in the filtered data
        total_count = len(filtered_data)
        hate_speech_count = filtered_data['hate_speech'].sum()
        percentage = (hate_speech_count / total_count) * 100

        # Display the percentage of hate speech and filtered data using Streamlit
        st.write(
            f"The percentage of hate speech in the filtered data is: {percentage:.2f}%")
        st.write(filtered_data)


def find_matching_images(suspect_folder, missing_person_image):
    # Load the missing person image
    missing_person = face_recognition.load_image_file(missing_person_image)
    missing_person_encoding = face_recognition.face_encodings(missing_person)[
        0]

    # Iterate over the suspect folder
    for file_name in os.listdir(suspect_folder):
        # Load the suspect image
        suspect_image = face_recognition.load_image_file(
            os.path.join(suspect_folder, file_name))
        suspect_encoding = face_recognition.face_encodings(suspect_image)

        if len(suspect_encoding) > 0:
            # Compare the face encodings
            match_results = face_recognition.compare_faces(
                [missing_person_encoding], suspect_encoding[0])
            if match_results[0]:
                # Display the matching image
                st.write(f"Match found: {file_name}")
                st.image(suspect_image, caption=file_name)


def main1():
    st.title("Missing Person Finder")

    # File upload for the missing person image
    st.subheader("Upload Missing Person Image")
    missing_person_image = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"])

    # Button to start finding the missing person
    if st.button("Find Missing Person"):
        # Check if the user has uploaded the missing person image
        if missing_person_image is not None:
            # Display the missing person image
            st.subheader("Missing Person Image")
            missing_person_image_pil = Image.open(missing_person_image)
            st.image(missing_person_image_pil)

            # Display the result images
            st.subheader("Result Images")
            suspect_folder = r"C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\images"
            find_matching_images(suspect_folder, missing_person_image)
        else:
            st.warning("Please upload the missing person image.")


def perform_sentiment_analysis(csv_file):
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row.get('Body')
            if text is not None:
                text = text.encode('ascii', 'ignore').decode('ascii')
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity

                if sentiment > 0:
                    sentiment_label = 'Positive'
                elif sentiment < 0:
                    sentiment_label = 'Negative'
                else:
                    sentiment_label = 'Neutral'

                st.write("Text:", text)
                st.write("Sentiment:", sentiment, "(", sentiment_label, ")\n")


def perform_sentiment_analysis_overall(csv_file):
    # Perform sentiment analysis on each text message
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as file:
        reader = csv.DictReader(file)

        for row in reader:
            text = row['Body']

            # Check if text is not None before creating TextBlob
            if text is not None:
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity

                if sentiment > 0:
                    positive_count += 1
                elif sentiment < 0:
                    negative_count += 1
                else:
                    neutral_count += 1

    # Display the overall sentiment analysis result
    st.subheader("Sentiment Analysis Result")
    st.write("Total Messages:", positive_count +
             negative_count + neutral_count)
    st.write("Positive Messages:", positive_count)
    st.write("Negative Messages:", negative_count)
    st.write("Neutral Messages:", neutral_count)


def main2():
    st.title("Sentiment Analysis")

    # Set the path to the CSV file
    csv_file = r'C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\sms\sms_log.csv'

   # Button to perform sentiment analysis
    if st.button("Perform Sentiment Analysis"):
        # Perform sentiment analysis and display the results
        perform_sentiment_analysis(csv_file)

    # Button to perform overall sentiment analysis
    if st.button("Perform Overall Sentiment Analysis"):
        # Perform overall sentiment analysis and display the results
        perform_sentiment_analysis_overall(csv_file)


def fetchs3():

    s3 = boto3.client('s3')
    bucket_name = 'hlosbucket125339-hlos'
    folder_prefix = 'public/'

    # Create the folder for downloads if it doesn't exist
    download_folder = 'user_mobile_storage'
    os.makedirs(download_folder, exist_ok=True)

    def download_folder_contents(bucket_name, folder_prefix, download_folder):
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
        for obj in response['Contents']:
            if obj['Key'] != folder_prefix:
                # Remove the folder prefix from the key
                relative_path = obj['Key'][len(folder_prefix):]
                file_path = os.path.join(download_folder, relative_path)
                if obj['Size'] == 0:
                    os.makedirs(file_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    s3.download_file(bucket_name, obj['Key'], file_path)
                    print(
                        f"File '{obj['Key']}' downloaded successfully to '{file_path}'.")

    download_folder_contents(bucket_name, folder_prefix, download_folder)


def display_call_info(data):
    df = pd.DataFrame(data, columns=['Number', 'Date', 'Duration', 'Type'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Duration'] = df['Duration'].astype(int)
    df['Type'] = df['Type'].astype(int)
    return df


def main7():
    # Load the data from the CSV file
    file_path = r'C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\call_logs\call_log.csv'
    data = pd.read_csv(file_path)

    # Call the function to display the call information
    call_info = display_call_info(data.values.tolist())

    # Create the Streamlit app
    st.title('Call Information')
    st.write(call_info)


def search_calls(phone_number, data):
    # calls = data[data['Number'] == phone_number]
    calls = data[data['Number'].astype(str) == phone_number]

    return calls


def main8():
    # Load the data from the CSV file
    file_path = r'C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\call_logs\call_log.csv'
    data = pd.read_csv(file_path)

    # Create the Streamlit app
    st.title('Call History Lookup')
    st.write('Enter a phone number to find the calls made by that number.')

    # Input field for phone number
    phone_number = st.text_input('Phone Number')

    if st.button('Search Calls'):
        if phone_number:
            calls = search_calls(phone_number, data)
            if not calls.empty:
                st.write('Call history for phone number:', phone_number)
                st.write(calls[['Date', 'Duration', 'Type']])
            else:
                st.write('No calls found for the given phone number.')
        else:
            st.write('Please enter a phone number.')


def get_image_info(image):
    image_info = {}

    # File format
    image_info['File Format'] = image.format

    # Image dimensions
    image_info['Image Dimensions'] = f"{image.width} x {image.height} pixels"

    # Color mode
    image_info['Color Mode'] = image.mode

    # Bit depth
    if image.mode == 'P':
        image_info['Bit Depth'] = image.info.get('bits', None)
    else:
        image_info['Bit Depth'] = image.mode

    # Compression (for JPEG)
    if image.format == 'JPEG':
        image_info['Compression'] = image.info.get('dpi', None)

    # Metadata (for Exif data)
    if hasattr(image, '_getexif'):
        exif_data = image._getexif()
        if exif_data:
            metadata = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == 'GPSInfo':
                    gps_info = {}
                    for key in value.keys():
                        sub_tag_name = GPSTAGS.get(key, key)
                        gps_info[sub_tag_name] = value[key]
                    metadata[tag_name] = gps_info
                else:
                    metadata[tag_name] = value
            image_info['Metadata'] = metadata

    return image_info


def format_gps_coordinates(coordinates):
    degrees = coordinates[0][0] / coordinates[0][1]
    minutes = coordinates[1][0] / coordinates[1][1]
    seconds = coordinates[2][0] / coordinates[2][1]
    return degrees + (minutes / 60) + (seconds / 3600)


def get_formatted_coordinates(gps_info):
    latitudes = gps_info.get('GPSLatitude')
    latitudes_ref = gps_info.get('GPSLatitudeRef')
    longitudes = gps_info.get('GPSLongitude')
    longitudes_ref = gps_info.get('GPSLongitudeRef')

    if latitudes and latitudes_ref and longitudes and longitudes_ref:
        lat = format_gps_coordinates(latitudes)
        lon = format_gps_coordinates(longitudes)
        lat = lat if latitudes_ref == 'N' else -lat
        lon = lon if longitudes_ref == 'E' else -lon

        return f"{lat:.6f}, {lon:.6f}"

    return None


def process_directory(directory_path):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter image files
    image_files = [file for file in files if file.lower().endswith(
        ('.png', '.jpeg', '.jpg'))]

    # Process each image file
    for file in image_files:
        file_path = os.path.join(directory_path, file)
        image = Image.open(file_path)

        st.subheader(f'Image: {file}')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Get and display the image information
        image_info = get_image_info(image)
        for key, value in image_info.items():
            if key == 'Metadata':
                st.write(f"- {key}:")
                metadata = value.copy()
                if 'GPSInfo' in metadata:
                    gps_info = metadata['GPSInfo']
                    for gps_key, gps_value in gps_info.items():
                        st.write(f"  - {gps_key}: {gps_value}")
                    del metadata['GPSInfo']
                for metadata_key, metadata_value in metadata.items():
                    st.write(f"  - {metadata_key}: {metadata_value}")
            else:
                st.write(f"- {key}: {value}")

        st.write('---')


def main9():
    # Create the Streamlit app
    st.title('Image Information')

    # Directory selection
    directory_path = r"C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\images"

    if st.button('Process Images'):
        if directory_path:
            process_directory(directory_path)


def profile():

    st.write("Device-Profile")
    st.write("Extraction is done!")
    st.title("Device: samsung-sm_a207f-R9GN1029DMJ")

    st.subheader("Call log info")
    # Load the data from the CSV file
    file_path = r'C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\call_logs\call_log.csv'
    data = pd.read_csv(file_path, error_bad_lines=False)
    num_entries = len(data)
    st.write(f"Number of entries in call log: {num_entries}")
    if st.button("get calls"):
        st.dataframe(data)

    st.subheader("Message info")
    # Load the data from the CSV file
    csv_file = r'C:\Users\APQA\Desktop\Hi\tech\vitall\user_mobile_storage\sms\sms_log.csv'
    datamsg = pd.read_csv(csv_file, error_bad_lines=False)
    num_entries_msg = len(datamsg)
    st.write(f"Number of entries in call log: {num_entries_msg}")

    if st.button("get messages"):
        st.dataframe(datamsg)


# Create a navbar
st.sidebar.title("Navigation")
nav_option = st.sidebar.radio("Go to", ("Profile report", "Person face recognition", "Sentiment analysis",
                              "Hate Speech analysis", "Voilence detection", "Call log", "Otps and messages", "Document reader", "Image Information"))

# Handle navbar options
if nav_option == "Profile report":
    # fetchs3()
    profile()
elif nav_option == "Person face recognition":
    main1()
elif nav_option == "Sentiment analysis":
    main2()
elif nav_option == "Hate Speech analysis":
    main3()
elif nav_option == "Otps and messages":
    main4()
elif nav_option == "Voilence detection":
    main5()
elif nav_option == "Document reader":
    main6()
elif nav_option == "Call log":
    main7()
    main8()
elif nav_option == "Image Information":
    main9()
