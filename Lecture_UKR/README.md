# UKRの実装

## 概要
古川研の基盤技術であるUnsupervised Kernel Regression（UKR）をスクラッチ実装する．

## 獲得スキルセット
ミニマム
- UKRのタスク設定を理解し，アルゴリズムの各ステップで何をしているか説明できる.
- ukr.pyを自動微分で穴埋めし，UKRを動かすことができる．
- 各種パラメータ（カーネル幅など）やデータを変えて観察できる．

スタンダード
- fとEの計算を行列計算を使わずにfor文で実装する．
- ukr.pyを手動微分で実装できる． ([参考資料1](https://github.com/furukawa-laboratory/rookies_workout_2021/blob/main/ukr/UKR_formula.pdf)を見ながらやってね！)
- 人工データを独自に生成できる．(data.pyに新しい関数を追加するイメージ！）
- 実装した二つのプログラムをクロステストできる．(cross_test.pyをいじればできるはず！)

エクストラ
- 手動微分を数式的に導くことができる．([参考資料３](https://github.com/furukawa-laboratory/KSE/blob/master/documents/folulary/kse%E5%85%AC%E5%BC%8F%E9%9B%86%E3%81%A8%E6%A4%9C%E7%AE%971.pdf)が参考になるよ！)
- 動物データやnipsなどの実データで結果を観察できる． (動物データは[somfのここから](https://github.com/furukawa-laboratory/somf/blob/master/libs/datasets/artificial/animal.py),nipsデータはwebから読み込んでね！)

## ファイルの説明 
- ukr.py: みんなが実装するファイル<br>
- data.py: 鞍型データ,螺旋型データ,sinデータを生成してくれるファイル<br>
- visualizer.py: アニメーションのviewerのファイル（今回はみんなは実装する必要がない想定だが，余裕があれば実装しても良い）<br>
- cross_test.py: 自分の実装が他の実装と差異がないかを確認するファイル．<br>

sampleフォルダは答えがあるので見ないでね！！！

## ミニマムまでの進め方
1.まずukr.pyの53行目までを埋めるか，一から実装するかして，以下の動画の状態（写像がぐちゃぐちゃなっている状態）まで実装する．（モデルの実装自体はうまく行っている可能性が高い．）

https://user-images.githubusercontent.com/63096349/165115960-06032316-68aa-47ec-a610-d26a4686818f.mp4

2.次に写像をメッシュ状に描画する．　ukr.pyの55-74, 102-104のコメントアウトを外して実装する．(ここまでできれば実装完了！)
ヒント：写像をメッシュ状にするには，潜在空間にメッシュ状の代表点をとって，そこからの写像を求めれば良いよね！

https://user-images.githubusercontent.com/63096349/165118034-179cfbf5-5c17-4e0c-a7d1-1ade0c2c77ed.mp4

3.パラメータやデータを変えて観察してみる．
- エポックを増やすと潜在変数はどうなる？
- カーネルの幅を大きくすると，小さくするとどうなる？
- 正則化をつけて観察してみよう．　正則化の種類によりどんな風に学習が止まる？
- 螺旋型データ, sinデータでも試してみよう．
- 学習がうまくいかないケースはどんなケース？

## みんなへのお願い（プルリクの完成度で評価するのでちゃんとまとめよう）
- 1,2,3が終わる度にプルリクにまとめて，先輩からレビューをもらう．
- ミニマムが終わったら，これからどのメニューをするのか（例えば手動微分にチャレンジします）など，一言プルリクのコメントに記す．
- （先に早く進むより，一つ一つちゃんと理解してこなすのが大事！）

## 参考資料
1. [UKRの数式集](https://github.com/furukawa-laboratory/rookies_workout_2021/blob/main/ukr/UKR_formula.pdf) 実装の時に参考になるよ．
2. [UKRの論文](https://www.sciencedirect.com/science/article/abs/pii/S0925231206004802)  1章と２章は勉強になるので読んでみてね！
3. [目的関数の微分式の導出](https://github.com/furukawa-laboratory/KSE/blob/master/documents/folulary/kse%E5%85%AC%E5%BC%8F%E9%9B%86%E3%81%A8%E6%A4%9C%E7%AE%971.pdf)  手動微分が導出される過程がわかるよ．
