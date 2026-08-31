# 自動微分

モデルが Sequential の型にはまらないとき (重み共有、カスタム損失、変わったアーキテクチャ) は、計算を直接組み立ててリバースモード自動微分に勾配を導出させます。エンジンは micrograd スタイル: n 次元テンソル上の `Node` の動的グラフで、define-by-run、使い捨てです。`Node` が持つ値は `Tensor` なので、要素ごとの演算はブロードキャストし、`MatMul` は行列のスタックを掛けます。リーフを作るところでは `Matrix` をゼロコピーの 2 次元ビューとして受け取るので、以下の 2 次元のコードは今までどおりです。

## Param、Input、Trainer

```go
w1 := autograd.Param(tensai.RandomMatrix(2, 8, rng))
b1 := autograd.Param(tensai.NewMatrix(1, 8))
w2 := autograd.Param(tensai.RandomMatrix(8, 1, rng))
trainer := autograd.NewTrainer(optim.NewAdam(0.05), w1, b1, w2)

for step := 0; step < 2000; step++ {
	loss := autograd.Input(x).MatMul(w1).AddRow(b1).Tanh().MatMul(w2).Sigmoid().MSELoss(y.Tensor())
	trainer.Step(loss) // backward + 更新 + 勾配ゼロ化。損失値を返す
}
```

`Param` は勾配を追跡して更新すべき行列を、`Input` はデータをラップします。手動で制御したければ部品はすべて公開されています: `loss.Backward()`, `p.Grad`, `autograd.ZeroGrads(params...)`。

## 使える演算

`MatMul` (バッチ対応、先頭軸はブロードキャスト), `Add`, `Sub`, `Mul`/`MulElem`, `Div`, `Scale`, `Neg`, `AddRow`, `T`/`Transpose`, `Reshape`, `Softmax` (最終軸), `Sum`, `Mean`, `SumAxis`/`MeanAxis`, `LayerNorm`, `Embed`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU`, `Exp`, `Log`, `MSELoss`, `SoftmaxCELoss`, `CrossEntropy`。要素ごとの演算は NumPy 流にブロードキャストし、勾配は引き伸ばされた軸ぶんだけ合計されて戻ります。

グラフはステップごとに動的に構築され、使い捨てです。形状の不一致はグラフ構築時に panic します。すべての演算の勾配はテストスイートで数値微分と照合されています。

## Tape によるバッファ再利用

グラフはステップごとに組み立てて捨てるので、既定では中間値と勾配のすべてを毎ステップ確保し直します。`Tape` はそれを再利用します。パラメータを一度 Bind して、ステップの最後に Reset するだけです:

```go
tape := autograd.NewTape()
tape.Bind(w1, b1, w2, b2) // op は親から tape を引き継ぐ

for step := 0; step < steps; step++ {
	trainer.Step(forward(x).MSELoss(y))
	tape.Reset() // このステップのバッファをプールに返す
}
```

規則は学習ループが元々守っているもの 1 つだけです: **`Reset` の後、終わったステップの値を読んではいけません** — `Value` も `Grad` もです。パラメータの値は再利用されない (tape が持つのは op が作ったものだけ) ので、学習した重みはいつでも保持して安全です。それ以外を残したいときは Reset の前にコピーしてください。`Tape` は並行利用できないので、学習ゴルーチンごとに 1 つ持たせます。

32 ステップを展開する `_example/charrnn` では、1 学習ステップの確保量が 22MB から 0.75MB に減り、実時間も約 1/4 短くなります。

同じ再利用は 1 段下でも使えます。`MatMulInto`, `MatMulTNInto`, `MatMulNTInto`, `AddInto`, `SubInto`, `MulInto`, `DivInto` は、行列に対する `DotInto` と同じように、呼び出し側が持つテンソルに書き込みます。

## グラフの可視化

`loss.ToDot()` は Graphviz DOT を返します (葉には `.Named("w1")` でラベルを付けられます):

```bash
go run ./_example/dot | dot -Tsvg > graph.svg
```

## リカレントネットワーク

`rnn.Cell` と `rnn.LSTMCell` は自動微分エンジンの上に組まれているので、シーケンスの展開はただの Go のループで、BPTT (通時的誤差逆伝播) は勝手についてきます:

```go
cell := rnn.NewLSTMCell(inSize, hidden, rng)
wOut := autograd.Param(tensai.RandomMatrix(hidden, numClasses, rng))
bOut := autograd.Param(tensai.NewMatrix(1, numClasses))
trainer := autograd.NewTrainer(optim.NewAdam(0.01), append(cell.Params(), wOut, bOut)...)

for step := 0; step < epochs; step++ {
	h, c := cell.InitState(batch)
	for _, x := range steps { // タイムステップごとに 1 つの (batch x inSize) 行列
		h, c = cell.Step(autograd.Input(x), h, c)
	}
	logits := h.MatMul(wOut).AddRow(bOut)
	trainer.Step(logits.CrossEntropy(labels)) // labels はクラス番号の []int
}
```

`_example/charrnn` は埋め込みのパブリックドメインテキストで文字レベル LSTM を学習し、再読込したパラメータからサンプルを生成します。

## Attention

`rnn.SelfAttention` は 1 つの `(seqLen x inSize)` シーケンスノードに作用します。`attn.Forward(x)` が学習可能な射影つきで `softmax(Q*K^T/sqrt(d))*V` を計算します。素の `rnn.Attention(q, k, v)` も公開されています。

## パラメータの保存

自動微分のパラメータは位置ベースに保存・復元します:

```go
autograd.SaveParamsFile("cell.json", cell.Params()...)
// 同じセルを組んでから
autograd.LoadParamsFile("cell.json", cell.Params()...)
```
