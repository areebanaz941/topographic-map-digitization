# Topographic Map Digitization and Feature Extraction
This project involves the digitization of a topographic map from an image, along with the extraction of key geographical features such as settlements, open areas, water bodies, and road networks. These features are extracted using Python libraries such as skimage, geopandas, and matplotlib, and are saved as a shapefile for further analysis and GIS applications.

## Project Overview
This project aims to automate the digitization process of geographical maps by:
- Loading a topographic map image.
- Applying various image processing techniques to extract key features such as:
  - Settlements
  - Open Areas
  - Water Bodies
  - Road Networks
- Saving the extracted features as vector data (shapefile format) for GIS analysis.

## Project Features
- **Image Loading**: Load the topographic map image.
- **Grayscale Conversion**: Convert the image to grayscale for better analysis.
- **Image Smoothing and Noise Reduction**: Apply Gaussian filters for noise reduction.
- **Thresholding**: Use Otsu’s method and custom thresholding values for different features.
- **Morphological Processing**: Perform closing operations to fill small gaps.
- **Feature Extraction**: Detect and extract contours to create polygon and line geometries.
- **Shapefile Creation**: Save the extracted features as an ESRI shapefile.
- **Visualization**: Visualize the digitized features on top of the original map using `matplotlib`.

## Technologies Used
- **Python Libraries**:
  - `numpy`: Numerical operations.
  - `matplotlib`: Visualization and plotting.
  - `skimage`: Image processing and feature extraction.
  - `geopandas`: Handling spatial data and vectorization.
  - `shapely`: Creating and manipulating geometric objects.
  - `Pillow (PIL)`: Image loading and manipulation.

## Project Workflow
### Image Loading:
- The project starts by loading a topographic map image using `PIL.Image.open()`.

### Image Preprocessing:
- The image is converted to grayscale using `skimage.color.rgb2gray()`.
- Histogram analysis is performed to study pixel intensity distribution.
- Gaussian filters are applied to smooth the image and reduce noise.

### Feature Extraction:
- Otsu's thresholding and manual thresholding techniques are applied to segment the image into different geographical features.
- Morphological closing operations are used to clean up small holes in detected features.
- Contours are detected using `skimage.measure.find_contours()` and converted into `shapely.Polygon` or `shapely.LineString`.

### Feature Visualization:
- The features are overlaid on the original image, each visualized with a unique color for easy identification.

### Shapefile Creation:
- The extracted features are stored as a `GeoDataFrame` and exported to an ESRI shapefile format.

## Results
- After running the script, you'll see a visual representation of the digitized map with different features like settlements, open areas, water bodies, and road networks, along with the saved shapefile for GIS usage.

### Study Area:
![image](https://github.com/user-attachments/assets/2964fa7c-7c5a-4afd-a4bc-dfe5065daddb)

![image](https://github.com/user-attachments/assets/65869018-b86a-49e2-871d-07b9d0291bd5)

![image](https://github.com/user-attachments/assets/9b408d7b-9c73-4c4d-9328-7cd7728704a4)

![image](https://github.com/user-attachments/assets/dad68fa0-1b13-45e4-844b-664fef90fcc6)

### Digitized Maps

![image](https://github.com/user-attachments/assets/08ffef4d-6012-4d79-81e1-4348cc0a17b7)

![image](https://github.com/user-attachments/assets/26b1b841-480d-47e6-bf04-013810eb65a2)


