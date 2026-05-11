# Convolutional Neural Network (CNN) Documentation

Yeh document aapko CNN ke workflow aur usmein use hone wale functions ko samajhne mein madad karega.

---

## 1. What is CNN?
CNN (Convolutional Neural Network) ek deep learning algorithm hai jo images ko process karne aur patterns (jaise edges, shapes, objects) pehchanne ke liye design kiya gaya hai.

---

## 2. Core Functions and Layers

### `to_categorical`
- **Kaam:** Yeh function integer labels (0, 1, 2...) ko **One-Hot Encoding** mein convert karta hai.
- **Example:** Agar aapke paas 3 classes hain (0, 1, 2), toh '1' convert hokar `[0, 1, 0]` ban jayega.
- **Import Fix:** 
  ```python
  from tensorflow.keras.utils import to_categorical
  ```

### `Conv2D` (Convolutional Layer)
- **Kaam:** Yeh layer image se features extract karti hai (jaise horizontal ya vertical lines). Yeh filters use karti hai jo image ke upar slide (convolve) hote hain.
- **Parameter:** `filters=32`, `kernel_size=(3,3)`, `activation='relu'`.

### `MaxPooling2D` (Pooling Layer)
- **Kaam:** Yeh image ke size ko chota (downsample) karti hai taki calculations kam ho sakein aur important features retain rahein. Yeh sabse bada value (Max) select karti hai pixel grid se.

### `Flatten`
- **Kaam:** CNN layers 2D (matrix) data output karti hain, lekin final output ke liye humein 1D (vector) data chahiye hota hai. `Flatten` usey ek single line mein convert kar deta hai.

### `Dense` (Fully Connected Layer)
- **Kaam:** Yeh normal Neural Network layer hai jahan har neuron pichli layer ke har neuron se connected hota hai. Last layer mein hum `Dense(classes, activation='softmax')` use karte hain classification ke liye.

### `Dropout`
- **Kaam:** Yeh overfitting rokne ke liye kuch neurons ko randomly deactivate kar deta hai training ke waqt.

---

## 3. Typical CNN Workflow

1.  **Data Preprocessing:** Images ka size set karna aur labels ko `to_categorical` se encode karna.
2.  **Model Building:** 
    - `Sequential` model initialize karna.
    - Convolution aur Pooling layers add karna.
    - Flattening karna.
    - Dense layers (Hidden + Output) add karna.
3.  **Compilation:** Optimizer (jaise 'adam') aur Loss function (jaise 'categorical_crossentropy') select karna.
4.  **Training:** `model.fit()` use karke model ko train karna.
5.  **Prediction:** `model.predict()` se nayi images check karna.

---

## Important Imports
Agar aap start kar rahe hain, toh yeh basic imports zaroori hain:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
```
