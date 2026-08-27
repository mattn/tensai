# 自動微分

モデルが Sequential の型にはまらないとき (重み共有、カスタム損失、変わったアーキテクチャ) は、計算を直接組み立ててリバースモード自動微分に勾配を導出させます。エンジンは micrograd スタイル: 行列上の `Node` の動的グラフで、define-by-run、使い捨てです。

## Param、Input、Trainer

```go
w1 := autograd.Param(tensai.RandomMatrix(2, 8, rng))
b1 := autograd.Param(tensai.NewMatrix(1, 8))
w2 := autograd.Param(tensai.RandomMatrix(8, 1, rng))
trainer := autograd.NewTrainer(optim.NewAdam(0.05), w1, b1, w2)

for step := 0; step < 2000; step++ {
	loss := autograd.Input(x).MatMul(w1).AddRow(b1).Tanh().MatMul(w2).Sigmoid().MSELoss(y)
	trainer.Step(loss) // backward + 更新 + 勾配ゼロ化。損失値を返す
}
```

`Param` は勾配を追跡して更新すべき行列を、`Input` はデータをラップします。手動で制御したければ部品はすべて公開されています: `loss.Backward()`, `p.Grad`, `autograd.ZeroGrads(params...)`。

## 使える演算

`MatMul`, `Add`, `Sub`, `MulElem`, `Scale`, `AddRow`, `T`, `Softmax`, `ReLU`, `Sigmoid`, `Tanh`, `Sum`, `Mean`, `MSELoss`, `SoftmaxCELoss`。

グラフはステップごとに動的に構築され、使い捨てです。形状の不一致はグラフ構築時に panic します。すべての演算の勾配はテストスイートで数値微分と照合されています。

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
	trainer.Step(logits.SoftmaxCELoss(labels))
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
