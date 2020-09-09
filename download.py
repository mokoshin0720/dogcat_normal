from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# APIキーの情報
api_key  = "8b7b518e15b029b576388acbc2546714"
api_secret = "e5bc4be07060f400"
wait_time = 1

# 保存フォルダの指定
vegetable_name = sys.argv[1]
savedir = "./" + vegetable_name

flickr = FlickrAPI(api_key, api_secret, format = "parsed-json")
result = flickr.photos.search(
    # 検索キーワード
    text = vegetable_name,
    # 取得するデータ数
    per_page = 500,
    # 検索するデータの種類
    media = "photos",
    # 検索時のデータの並び順
    sort = "relevance",
    # UIコンテンツの非表示
    safe_search = 1,
    # 取得したいデータ
    extras = "url_q, licence"
)

# 結果を表示する
photos = result["photos"]
pprint(photos)

for i, photo in enumerate(photos["photo"]):
    url_q = photo["url_q"]
    filepath = savedir + "/" + photo["id"] + ".jpg"
    
    # ファイルがすでにあればスキップする
    if os.path.exists(filepath): continue
    # URLから画像ファイルを保存
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)