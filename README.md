# Random Dot Stereo Club

For the love of random dots and stereo vision.

Blenderで作った机上シーンの、ドット投影付きステレオデータセットです。D435、D555、合成ドット投影器を追加したOAK-Dモデルで、同じ配置を撮像しています。透明な物体も含みます。

## サンプル

同じ評価シーンを3機種で撮像した例です。左右IR、正解視差、LAS2-Hの推定視差、絶対誤差、左右からの可視性を並べています。

| D435 | D555 | OAK-D + synthetic projector |
| --- | --- | --- |
| [プレビュー](samples/D435.png) | [プレビュー](samples/D555.png) | [プレビュー](samples/OAK-D-active.png) |

![OAK-Dの左右IR画像と視差の比較](samples/OAK-D-active.png)

## データセット

画像は1280×720です。各サンプルに左右IR、RGB、左IR・RGB視点の正解視差PFM、RGBに整列したインスタンスマスク、視差の有効性・ステレオ可視性・透明材を示すマスク、設定JSONが含まれます。左右IRはシーン内で共通の投影パターンを観測し、再生成に使えるシードを設定JSONに記録しています。

評価セットは学習用とは別に、未使用の物体形状、屋内HDRI、PBRテクスチャを使って作成しています。正解視差は各視点で最初に当たった表面に対応します。距離では切り分けず、対応視点からの可視性は別マスクで提供します。

本体のダウンロード先はZenodoで公開後、ここにDOIを掲載します。利用条件は公開時にZenodoのレコードに明記します。

## カメラパラメータ

[D435](calibration/D435.json) · [D555](calibration/D555.json) · [OAK-D-active](calibration/OAK-D-active.json)

JSONに1280×720の内部行列 `K`、左右IRの基線、左IRから各カメラへの外部変換、投影器の位置と画角を記載しています。座標軸は右が +X、下が +Y、前が +Z。3D座標と並進はメートル、内部行列はピクセルです。画像カメラは歪みなしのピンホールモデルです。値は合成データ生成時の設定で、実機個体の工場校正値は使用していません。OAK-D-activeの投影器は合成用に追加した仕様です。

左IR視差 `d` から深度を求めるには `Z = fx_left × baseline / d` を使い、`Z × inv(K_left) × [u, v, 1]ᵀ` で左IRの3D座標に戻します。RGBへの投影は `X_rgb = R × X_left + t_m` で座標を変換し、`K_rgb` で投影します。`disparity_rgb.pfm` はRGB画素上での左右IR視差で、RGB独自の焦点距離で定義した視差ではありません。

## PyTorchで視差から深度画像を作る

[使用例](examples/disparity_to_depth.py)は、校正JSONを読み、仮のネットワーク関数 `predict_disparity(left, right)` が返す視差から、左IR視点の深度画像を作ります。仮の関数はどの画素にも32 pxを返すため、深度画像は正面を向いた平面になります。

```bash
python examples/disparity_to_depth.py --calibration calibration/D435.json --output depth_m.npy
```

自分のモデルを使う場合は、左右の入力テンソルをモデル用に前処理し、`predict_disparity` の中身をモデルの推論に差し替えます。戻り値は入力と同じデバイスにある `[B, H, W]` の左基準視差です。ネットワークのパディングを外し、サイズ変更した場合は画像サイズと視差のpxスケールを校正時の解像度に戻します。その後は共通の `disparity_to_depth` で変換できます。

CUDAが使える環境では、推論用のテンソル作成から深度変換までGPU上で処理します。`--device cpu` または `--device cuda` で明示指定できます。保存結果は `[H, W]` のfloat32 NumPy配列で、単位はメートルです。ゼロ・負値・非有限の視差はNaNになり、1 px未満でも正の視差は変換します。


## 🍺

もしこのデータセットが役に立ったなら、どこかで会ったときにビール一杯おごってね。  
If this dataset helped you and we ever meet, you can buy me a beer.
