import requests
from bs4 import BeautifulSoup

# URL halaman yang ingin di-scrape
url = "https://disparda.baliprov.go.id/category/daya-tarik-wisata/karangasem/"  # Ganti dengan URL yang sesuai

# Menonaktifkan verifikasi SSL
response = requests.get(url, verify=False)

# Menampilkan respons
print(response.text)