import os
from datetime import datetime, timezone

os.makedirs("dist", exist_ok=True)

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>自動生成ページ</title>
</head>
<body>
  <h1>GitHub Actionsで生成されたページ</h1>
  <p>生成日時: {now}</p>
</body>
</html>
"""

with open("dist/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("dist/index.html を生成しました")
