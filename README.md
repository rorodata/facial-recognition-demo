# facial-recognition-demo

## Testing the API:

```
import firefly
import binascii

api = firefly.Client("https://facial-recognition-demo.rorocloud.io/")

image_data = open('./test_image.jpg', 'rb').read()
image_data = binascii.b2a_base64(image_data)
image_data = str(image_data, encoding='utf-8')

api.predict(ascii_image_data = image_data)
```



