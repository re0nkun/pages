import datetime

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>自動生成ページ</title>
</head>
<body>
<h1>Hello from GitHub Actions</h1>
<p>生成日時: {now}</p>
</body>
</html>
"""

with open("output/index.html", "w", encoding="utf-8") as f:
    f.write(html)
