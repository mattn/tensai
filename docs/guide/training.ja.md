# モデルの学習

## Sequential のワークフロー

レイヤーを積んで、`Compile` → `Fit` (または `FitStep`) → `Predict`:

```go
net := model.NewSequential()
net.Add(layer.NewDense(8))
net.Add(&layer.Tanh{})
net.Add(layer.NewDense(1))
net.Add(&layer.Sigmoid{})

net.Compile(2, loss.MeanSquaredError{}, optim.NewAdam(0.05))
net.Fit(inputs, targets, 5000)   // フルバッチで 5000 エポック

pred, _ := net.Predict(inputs)
```

`Compile(inputCols, loss, optimizer)` が各レイヤーを初期化し、列数をスタックに通します。`Fit` はフルバッチのエポックを回し、`FitStep(input, target)` はちょうど 1 回の順伝播/逆伝播/更新を実行して損失を返します — ミニバッチ学習の構成要素です。

## データセット

`Dataset` は入力とターゲットをペアにして、シャッフル、分割、標準化、バッファ再利用のミニバッチ反復を提供します:

```go
ds, _ := dataset.New(inputs, targets)
ds.Shuffle(rng)
train, test, _ := ds.Split(0.2)          // ビュー、コピーなし
mean, std := train.Standardize()         // train でフィットして...
test.StandardizeWith(mean, std)          // ...test に適用

for epoch := 0; epoch < epochs; epoch++ {
	train.Batches(32, rng, func(in, tgt *tensai.Matrix) error {
		_, err := net.FitStep(in, tgt)
		return err
	})
}
```

`Split` はコピーのないビューを返し、`Batches` はバッチバッファを反復をまたいで再利用するので、内側のループにアロケーションが入りません。

## 畳み込みモデル

```go
net := model.NewSequential()
net.Add(layer.NewConv2D(8, 3, 1, 1)) // outC, kernel, stride, pad
net.Add(&layer.ReLU{})
net.Add(layer.NewMaxPool2D(2))
net.Add(layer.NewDense(64))
net.Add(layer.NewBatchNorm())
net.Add(layer.NewLeakyReLU(0.01))
net.Add(layer.NewDropout(0.3))
net.Add(layer.NewDense(10))

// 入力の形状は一度だけ書く。空間形状がスタックを流れるので
// conv / pool レイヤーは自分の入力サイズをそこから受け取る
net.CompileImage(layer.Image{H: 28, W: 28, C: 1}, loss.SoftmaxCrossEntropy{}, optim.NewAdamW(0.001, 0.01))
net.Fit(inputs, targets, 10)
```

各入力行はチャンネル優先でフラット化した画像です (`index = (channel*height + y)*width + x`)。`Dropout` と `BatchNorm` は `Fit`/`FitStep` 内では自動的に学習モード、`Predict` 内では推論モードになります。

## 保存と読み込み

`Save`/`Load` (と `SaveFile`/`LoadFile`) は、学習済み Sequential のパラメータを BatchNorm の移動統計量も含めて JSON でラウンドトリップします:

```go
net.SaveFile("model.json")

// 後で: 同じアーキテクチャを組んで Compile してから
net.LoadFile("model.json")
```

アーキテクチャ自体はシリアライズされません — 同じレイヤースタックを再構築して `Compile` してから読み込んでください。自動微分のパラメータ (RNN/LSTM/attention セル) は `SaveParams`/`LoadParams` で位置ベースに保存します — [自動微分](autograd.md)参照。

Go の外へのデプロイには TFLite / ONNX エクスポートがあります — [モデルフォーマット](../formats.md)参照。

## 低アロケーション学習

レイヤーは順伝播/逆伝播のスクラッチバッファを学習ステップをまたいで再利用するので、MLP の 1 ステップは約 29 アロケーションで済み、GC は学習ループの外に留まります。`Predict` は常に新しくアロケートした結果を返すので、予測は保持しても安全です。
