import requests
url = "https://susya.onrender.com"
#imgdata = base64 string of image
r = requests.post(url,json = {"image":imgdata})
print(r.text.strip())