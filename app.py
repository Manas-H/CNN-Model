from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import io
import os
import base64

app = Flask(__name__)

# Load a model
model = YOLO('E:/Final-Year-Project/CNN-Model/CNN model/best.pt')  # load a custom model


@app.route('/', methods=['GET', 'POST'])
def upload_and_display():
    selected_image_base64 = None  # Initialize with None
    output_images_base64 = None
    
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'image' in request.files:
            image_file = request.files['image']
            
            if image_file:
                # Specify the directory and filename for saving the uploaded image
                upload_directory = 'E:/Final-Year-Project/CNN-Model/UI/Test'
                upload_path = os.path.join(upload_directory, 'uploaded_image.png')

                # Save the uploaded image to the specified path
                image_file.save(upload_path)

                # Load and predict with the model using the uploaded image
                img = Image.open(upload_path)
                results = model(upload_path, conf=0.40, iou=0.50)  # Use the uploaded image
                
                # Perform non-maximum suppression on YOLO prediction

                # Iterate over filtered results and visualize them
                output_images = []
                for r in results:
                    prob = r.probs
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    output_images.append(im)

                # Convert output images to base64 for displaying in HTML
                output_images_base64 = []
                for output_image in output_images:
                    buffered = io.BytesIO()
                    output_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    output_images_base64.append(img_str)
                
                # Convert the selected image to base64
                selected_image_buffered = io.BytesIO()
                img.save(selected_image_buffered, format="PNG")
                selected_image_base64 = base64.b64encode(selected_image_buffered.getvalue()).decode()

    return render_template('index.html', selected_image=selected_image_base64, output_images=output_images_base64)

if __name__ == '__main__':
    app.run(debug=True)