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

## Sequential を自動微分エンジンで学習する

`Fit` は各層の Forward と Backward を回します。`net.Graph()` は同じモデルを autograd グラフとして組み立てます:

```go
g, err := net.Graph()
trainer := autograd.NewTrainer(optim.NewAdam(0.01), g.Params()...)
tape := autograd.NewTape()
tape.UseDevice(dev) // 任意。GPU ガイド参照
tape.Bind(g.Params()...)

for step := 0; step < steps; step++ {
	loss, _ := g.Loss(g.Forward(autograd.Input(x)), y)
	trainer.Step(loss)
	tape.Reset()
}
g.Sync() // デバイス上の重みを層に書き戻す
```

パラメータは層が持っている行列そのものなので、グラフで学習することはモデルを学習することです。`Predict` も `Save` もエクスポートもそのまま動き、グラフの順伝播は `Predict` と float32 の丸め誤差の範囲で一致します。増えるのは GPU です — グラフは tape が送った先で走りますが、手書きのスタックはそうはいきません。

CPU では両者は数 % の差に収まります (`go test -bench Step ./model`)。最初からそうだったわけではなく、突き合わせてみて層スタック側の取りこぼしが 2 つ見つかり、直した結果 1024 幅のモデルの 1 ステップが 1.5 倍速くなりました。確保量は層スタックの方が少ない (層ごとのスクラッチを使い回すのに対し、グラフは中間値を tape から取る) ので、既定は引き続き層スタックです。

すべての層にグラフ形があります。Dropout と BatchNorm の学習時/推論時の切り替えは `SetTraining` で、層スタックにおける `Fit` と `Predict` に相当します。

BatchNorm は一言添える価値があります。グラフ形は定義そのまま (バッチの平均と分散を取り、正規化し、スケールしてシフトする) で、そこからエンジンが導く勾配は層が持つ手書きの勾配と要素単位で一致します。running estimates の更新も同じです — どちらでも順伝播の副作用だからです。

## 低アロケーション学習

レイヤーは順伝播/逆伝播のスクラッチバッファを学習ステップをまたいで再利用するので、MLP の 1 ステップは約 29 アロケーションで済み、GC は学習ループの外に留まります。`Predict` は常に新しくアロケートした結果を返すので、予測は保持しても安全です。
