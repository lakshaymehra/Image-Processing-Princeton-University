# Image Processing Assignment :

### Note: I have used Python for this assignment and I have avoided using libraries except for basic reading, writing, and displaying.

## Usage Guide:

### File Descriptions:

* ##### 'image_features_implementations.py' :
    contains the code to implement all the required features.

### Implementing Features:

* ##### To change the brightness of an image, use:
```bash
 python .\image_features_implementations.py --image_path 'input/princeton_small.jpg' --feature 'brightness' --output_image_path 'output/output.jpg' --beta 0.5
```
Change the value of the image_path, output_image_path, as well as beta according to your requirements to change the brightness of the input image.

* ##### To change the contrast of an image, use:
```bash
python .\image_features_implementations.py --image_path 'input/c.jpg' --feature 'contrast' --output_image_path 'output/output.jpg' --alpha 0.5
```
Change the value of the image_path, output_image_path, as well as alpha according to your requirements to change the contrast of the input image.

* ##### To blur an image, use:
```bash
python .\image_features_implementations.py --image_path 'input/princeton_small.jpg' --feature 'blur' --output_image_path 'output/output.jpg' --sigma 2
```
Change the value of the image_path, output_image_path, as well as sigma according to your requirements to blur the input image.

* ##### To sharpen an image, use:
```bash
python .\image_features_implementations.py --image_path 'input/princeton_small.jpg' --feature 'sharpen' --output_image_path 'output/output.jpg'
```
Change the value of the image_path, output_image_path according to your requirements.

Note: I have used the following kernel to sharpen the image :

      [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]

* ##### To detect edges in an image, use:
```bash
python .\image_features_implementations.py --image_path 'input/princeton_small.jpg' --feature 'edge_detection' --output_image_path 'output/output.jpg'
```
Change the value of the image_path, output_image_path according to your requirements.

Note: I have used the following kernel to detect the edges of the image :

      [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

* ##### To resize an image, use:
```bash
python .\image_features_implementations.py --image_path 'input/princeton_small.jpg' --feature 'resize' --output_image_path 'output/output.jpg' --new_image_height 100 --new_image_width 200
```
Change the value of the image_path, output_image_path, as well as new image's height and width according to your requirements to resize the input image.

* ##### To point rescale an image, use:
```bash
python .\image_features_implementations.py --image_path 'input/princeton_small.jpg' --feature 'scale_point' --output_image_path 'output/output.jpg' --new_image_height 100 --new_image_width 200
```
Change the value of the image_path, output_image_path, as well as new image's height and width according to your requirements to rescale the input image.

* ##### To bilinear rescale an image, use:
```bash
python .\image_features_implementations.py --image_path 'input/princeton_small.jpg' --feature 'scale_bilinear' --output_image_path 'output/output.jpg' --new_image_height 100 --new_image_width 200
```
Change the value of the image_path, output_image_path, as well as new image's height and width according to your requirements to rescale the input image.

* ##### To Gaussian rescale an image, use:
```bash
python .\image_features_implementations.py --image_path 'input/princeton_small.jpg' --feature 'scale_gaussian' --output_image_path 'output/output.jpg' --kernel_radius 1
```
Change the value of the image_path, output_image_path, as well as kernel radius according to your requirements to rescale the input image.

Note : I use the following formula to design the filter size :

      kernel_size =  2 * kernel_radius + 1
So, kernel_radius = 1 generates a 3x3 kernel to rescale the image,
    kernel_radius = 2 generates a 5x5 kernel, and so on...

* ##### To generate a Composite Image, use:
```bash
python .\image_features_implementations.py --feature 'composite' --foreground_image_path 'input/comp_foreground.jpg' --background_image_path 'input/comp_background.jpg' --mask_image_path 'input/comp_mask.jpg' --output_image_path 'output/output.jpg'
```
Change the value of the image_path, output_image_path, foreground_image_path, background_image_path as well as mask_image_path according to your requirements.

### Contact :
For any question, please contact
```
Lakshay Mehra: mehralakshay2@gmail.com
```

