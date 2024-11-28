import requests
url = "https://hydrovista.onrender.com"
#imgdata = base64 string of image
r = requests.post(url,json = {"image":imgdata})
print(r.text.strip())
