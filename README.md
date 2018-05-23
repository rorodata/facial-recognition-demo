# face-recognition-demo-tl

<p>
<img style="display: inline;" src="images/new_image.jpg">
</p>

See the [notebook](Facial-Recognition.ipynb) to get started.


## Testing the API:

```
import roro
import binascii

api = roro.Client("https://facial-recognition-demo.rorocloud.io/")

image_data = open('./test_image.jpg', 'rb').read()
image_data = binascii.b2a_base64(image_data)
image_data = str(image_data, encoding='utf-8')

new_image = api.tag_faces(ascii_image_data = image_data)

new_image = binascii.a2b_base64(new_image)
new_image = Image.open(fp=io.BytesIO(new_image))
new_image = cv.cvtColor(np.array(new_image), cv.COLOR_RGB2BGR)

cv.imwrite('new_image.jpg', new_image)
```




