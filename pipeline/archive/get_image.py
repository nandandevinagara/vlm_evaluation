import requests
import os

# The correct raw image URL
image_url = "https://raw.githubusercontent.com/yavuzceliker/sample-images/main/images/image-1.jpg"
#image_url = 'https://github.com/yavuzceliker/sample-images/blob/main/images/image-1.jpg'
# Define the local filename to save the image
# You can extract the filename from the URL or give it a custom name
filename = os.path.basename(image_url) # Extracts 'image-1.jpg'
# Or, set a custom filename:
# filename = "downloaded_image_from_github.jpg"

print(f"Attempting to download image from: {image_url}")
print(f"Saving to: {filename}")

try:
    # Send a GET request to the image URL
    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)

    # Open the file in binary write mode ('wb')
    with open(filename, 'wb') as f:
        # Write the content of the response (which is the binary image data) to the file
        f.write(response.content)

    print(f"Image '{filename}' downloaded successfully!")

except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")
except IOError as e:
    print(f"Error saving image to file: {e}")
    