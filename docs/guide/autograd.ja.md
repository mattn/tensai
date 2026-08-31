# 自動微分

モデルが Sequential の型にはまらないとき (重み共有、カスタム損失、変わったアーキテクチャ) は、計算を直接組み立ててリバースモード自動微分に勾配を導出させます。エンジンは micrograd スタイル: `Node` の動的グラフで、define-by-run、使い捨てです。

`Node` が持つ値は n 次元の `Tensor` です。つまり `(rows, cols)` の行列に効く演算がそのまま `(batch, sequence, model)` の活性にも効きます。要素ごとの演算は NumPy 流にブロードキャストし、`MatMul` は行列のスタックをまとめて掛けます。リーフを作るところでは `Matrix` をゼロコピーの 2 次元ビューとして受け取るので、2 次元のコードは今までどおりに読めます。

## Param、Input、Trainer

```go
w1 := autograd.Param(tensai.RandomMatrix(2, 8, rng))
b1 := autograd.Param(tensai.NewMatrix(1, 8))
w2 := autograd.Param(tensai.RandomMatrix(8, 1, rng))
trainer := autograd.NewTrainer(optim.NewAdam(0.05), w1, b1, w2)

for step := 0; step < 2000; step++ {
	loss := autograd.Input(x).MatMul(w1).AddRow(b1).Tanh().MatMul(w2).Sigmoid().MSELoss(y.Tensor())
	trainer.Step(loss) // 逆伝播 + 更新 + 勾配クリア、損失値を返す
}
```

`Param` は勾配を追跡して更新する値、`Input` はデータを包みます。どちらも `*tensai.Matrix` と `*tensai.Tensor` の両方を受け取ります。行列は同じ配列を共有する 2 次元テンソルビューになるので、行列から作ったパラメータはその行列を更新し続けます。

手動で回したいときの部品も公開されています: `loss.Backward()`、`p.Grad`、`autograd.ZeroGrads(params...)`。`Backward` は要素が 1 つのノードから始まり、`Grad` に**加算**します。学習ステップの最後に勾配をクリアするのはそのためです (`Trainer.Step` がやってくれます)。

ノードの `Value` と `Grad` は `*tensai.Tensor` です。`Shape()` で形状、`Matrix()` で 2 次元ノードの行列ビュー、`Scalar()` で 1 要素ノードの値が取れ、`Named("w1")` は `ToDot` 用のラベルです。

## 演算

| 演算 | 形状 |
| --- | --- |
| `MatMul(o)` | `(…, m, k) * (…, k, n)` → `(…, m, n)`。先頭の軸はブロードキャストするので、2 次元の重み 1 枚がバッチ全体に効きます |
| `Add`, `Sub`, `Mul` (`MulElem`), `Div` | 要素ごと、NumPy 流のブロードキャスト |
| `Scale(s)`, `Neg()` | スカラー倍 |
| `AddRow(row)` | `(m, n) + (1, n)`。`Add` が元々ブロードキャストするので、意図を明示するだけの形です |
| `T()`, `Transpose(perm...)` | 最後の 2 軸を入れ替え / 全軸を並べ替え |
| `Reshape(shape...)` | 1 つだけ `-1` を書けます。バッファは共有します |
| `Softmax()` | 最終軸に対して |
| `LayerNorm(gain, bias, eps)` | 最終軸に対して。`gain` と `bias` は特徴量ごとに 1 要素で、`nil` でも学習対象でも構いません |
| `Embed(ids, shape...)` | `(vocab, d)` のテーブルと `len(ids)` 個の添字 → `shape…, d`。同じ id が複数あれば逆伝播で加算されます |
| `Sum()`, `Mean()` | 全体を 1 要素に集約 |
| `SumAxis(axis, keepDims)`, `MeanAxis(axis, keepDims)` | 1 軸を集約。負の軸は末尾から数えます |
| `ReLU()`, `LeakyReLU(a)`, `Sigmoid()`, `Tanh()`, `GELU()`, `Exp()`, `Log()` | 要素ごと |
| `MSELoss(target)`, `SoftmaxCELoss(target)`, `CrossEntropy(labels []int)` | スカラー損失。交差エントロピーは最終軸をクラスとして読みます |

グラフはステップごとに動的に組み立てる使い捨てです。形状の不一致はエラーではなく構築時の panic になります。エラーを返すとチェーンが書けなくなりますし、形状違いはプログラミングのミスだからです。すべての演算の勾配は、テストで数値微分と突き合わせて検証しています。

## ブロードキャスト

要素ごとの演算は末尾の軸を揃え、長さ 1 の軸を引き伸ばします。`(1, 1, d)` のバイアスが `(batch, seq, d)` の活性に足せるのはこのためです。勾配は逆向きに同じ規則に従い、**引き伸ばした軸ぶんだけ合計されて戻ります**。バイアスも、特徴量ごとの `LayerNorm` の gain も、バッチで共有した重みも、それを使った全位置の寄与を特別扱いなしに集められるのはこの一点のおかげです。

## Attention の組み立て

マルチヘッド attention は reshape と transpose とバッチ積 2 回です:

```go
// x は (batch, seq, model)、wq, wk, wv, wo は (model, model)。
heads := func(t *autograd.Node) *autograd.Node {
	// (batch, seq, model) -> (batch, head, seq, headDim)
	return t.Reshape(batch, seq, nHeads, headDim).Transpose(0, 2, 1, 3)
}
q, k, v := heads(x.MatMul(wq)), heads(x.MatMul(wk)), heads(x.MatMul(wv))

// 全シーケンス・全ヘッドのスコア (batch, head, seq, seq) を一度に。
att := q.MatMul(k.T()).Scale(1 / float32(math.Sqrt(headDim))).Add(mask).Softmax()

y := att.MatMul(v).Transpose(0, 2, 1, 3).Reshape(batch, seq, model).MatMul(wo)
```

`mask` は `(1, 1, seq, seq)` の定数 `Input` で、対角と下側が `0`、上側が `math.Inf(-1)` です。バッチとヘッドにブロードキャストされ、softmax が未来の位置に重みを与えなくなります。`_example/tinygpt` はこれをそのまま使った文字単位の transformer (pre-norm ブロック、GELU の feed-forward、次文字の交差エントロピー) で、1 分ほどで 1 ページ分のテキストを覚えます。

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

規則は学習ループが元々守っているもの 1 つだけです: **`Reset` の後、終わったステップの値を読んではいけません** — `Value` も `Grad` もです。パラメータの値は再利用されない (tape が持つのは op が作ったものだけ) ので、学習した重みはいつでも保持して安全です。それ以外を残したいときは `Clone` でコピーしてから Reset してください。`Tape` は並行利用できないので、学習ゴルーチンごとに 1 つ持たせます。

32 ステップを展開する `_example/charrnn` では、1 学習ステップの確保量が 22MB から 0.75MB に減り、実時間も約 1/4 短くなります。

同じ再利用は 1 段下でも使えます。`MatMulInto`, `MatMulTNInto`, `MatMulNTInto`, `AddInto`, `SubInto`, `MulInto`, `DivInto` は、行列に対する `DotInto` と同じように、呼び出し側が持つテンソルに書き込みます。

## グラフの可視化

`loss.ToDot()` は Graphviz の DOT を返します (リーフには `.Named("w1")` でラベルを付けます):

```bash
go run ./_example/dot | dot -Tsvg > graph.svg
```

## リカレントネットワーク

`rnn.Cell` と `rnn.LSTMCell` は自動微分エンジンの上に乗っているので、系列の展開はただの Go のループで、BPTT は自動で付いてきます:

```go
cell := rnn.NewLSTMCell(inSize, hidden, rng)
wOut := autograd.Param(tensai.RandomMatrix(hidden, numClasses, rng))
bOut := autograd.Param(tensai.NewMatrix(1, numClasses))
trainer := autograd.NewTrainer(optim.NewAdam(0.01), append(cell.Params(), wOut, bOut)...)

for step := 0; step < epochs; step++ {
	h, c := cell.InitState(batch)
	for _, x := range steps { // 1 時刻あたり (batch x inSize) の行列 1 枚
		h, c = cell.Step(autograd.Input(x), h, c)
	}
	logits := h.MatMul(wOut).AddRow(bOut)
	trainer.Step(logits.CrossEntropy(labels)) // labels はクラス番号の []int
}
```

`_example/charrnn` は埋め込んだパブリックドメインのテキストで文字単位の LSTM を学習し、読み直したパラメータから文章を生成します。

`rnn.SelfAttention` はシングルヘッド・1 系列版です。`attn.Forward(x)` が `(seqLen, inSize)` のノードに対して学習済み射影付きの `softmax(Q*K^T/sqrt(d))*V` を計算し、素の `rnn.Attention(q, k, v)` も公開されています。バッチとヘッドが要るときは上のブロックを書いてください。

## パラメータの保存

自動微分のパラメータは順番で保存・復元します:

```go
autograd.SaveParamsFile("cell.json", cell.Params()...)
// 同じセルを組み立ててから
autograd.LoadParamsFile("cell.json", cell.Params()...)
```

任意のランクのパラメータが往復します。2 次元のものは以前のチェックポイントと同じエンコーディングを保つので、エンジンが n 次元になる前に書いたファイルもそのまま読めます。
