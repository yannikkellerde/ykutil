import base64
from mimetypes import guess_type

from ykutil import AzureModelWrapper


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


# Example usage
image_path = "/mnt/c/Users/ykeller/Pictures/buff_llama.png"
data_url = local_image_to_data_url(image_path)
print("Data URL:", data_url)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                },
            },
        ],
    }
]

model = AzureModelWrapper()

print(model.complete(messages))
