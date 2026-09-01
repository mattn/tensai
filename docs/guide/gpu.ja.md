# GPU (WebGPU)

`-tags wgpu` (linux, macOS, Windows) でビルドすると、CPU 版 `MatMul` と同じ形状・ブロードキャスト意味論の実験的 GPU バックエンドが有効になります。cgo は使いません: バインディングは [wgpu-native](https://github.com/gfx-rs/wgpu-native) 共有ライブラリを `ebitengine/purego` 経由で実行時にロードします (linux/macOS は `dlopen`、Windows は `LoadLibrary`)。wgpu-native は Linux で Vulkan、Windows で Vulkan か D3D12、macOS で Metal を選ぶので、AMD、Intel、Apple、NVIDIA の GPU がすべて動きます — lavapipe のような CPU Vulkan 実装も動き、GPU のないマシンでのテストはそれで回っています。

## セットアップ

**v22.1.0.5** のリリースバイナリ (このバインディングが対象とする C API) をダウンロードし、ローダーが見つかる場所に置くか `TENSAI_WGPU_LIB` で指定します:

```bash
curl -sLO https://github.com/gfx-rs/wgpu-native/releases/download/v22.1.0.5/wgpu-linux-x86_64-release.zip
unzip wgpu-linux-x86_64-release.zip -d wgpu
TENSAI_WGPU_LIB=$PWD/wgpu/lib/libwgpu_native.so go test -tags wgpu ./...
```

Windows では同じリリースの `wgpu-windows-x86_64-msvc-release.zip` を取り、中の `wgpu_native.dll` を変数で指定します (`PATH` 上か実行ファイルの隣にある `wgpu_native.dll` は変数なしで見つかります):

```powershell
$env:TENSAI_WGPU_LIB="$PWD\wgpu\lib\wgpu_native.dll"
go run -tags wgpu ./_example/wgpu
```

ビルドタグなしでは `gpu.Open` がエラーを返すだけで、他には何も変わりません。

## 基本的な使い方

```go
dev, err := gpu.Open() // GPU やライブラリがなければきれいに失敗する
if err != nil { /* tensai.MatMul にフォールバック */ }
defer dev.Close()
fmt.Println(dev.Name()) // 例: "AMD Radeon 780M (integrated)"
out, err := dev.MatMul(a, b)
```

iGPU と dGPU の両方があるマシンでは希望を渡せます: `gpu.Open(gpu.LowPower)` は iGPU へ、`gpu.HighPerformance` は dGPU へ寄せます (ヒントです — アダプタが 1 つならそれが返ります)。

## 常駐バッファ

`dev.MatMul(a, b)` は Upload → MatMul → Download → Free の省略形です。バッファを GPU に常駐させれば、重みはバスを 1 回しか渡らず、中間結果はデバイスを離れません:

```go
gw, _ := dev.Upload(w)              // 重みは 1 回だけアップロード
defer gw.Free()                     // GPU メモリは GC されない
gx, _ := dev.Upload(x)
h, _ := gx.MatMul(gw)               // 自由にチェーン。ホストには触れない
out, _ := h.MatMul(gw2)
result, _ := out.Download()         // 最後に 1 回だけ読み戻す
```

常駐が最も効くのは転送のたびに PCIe を渡るディスクリート GPU です。共有メモリの iGPU では効果は小さく、主に中間結果の読み戻しを省ける分になります。

## デバイス上の attention

MatMul の他に、常駐テンソルは `MatMulT`、インプレースの `Scale`、最終軸の行並列 `Softmax` をサポートします — シングルヘッド attention を GPU 上で完結させるのに十分です:

```go
out, _ := gq.Attention(gk, gv)                 // softmax(q@k^T/sqrt(d))@v、ホスト往復なし
out, _ = gq.MultiHeadAttention(gk, gv, heads)  // パックされた (batch, seq, heads*dh) レイアウト
```

マルチヘッド attention はストライド付きカーネルでパックレイアウトから各ヘッドを切り出すので、permute は一切実体化されません。causal 版 (`CausalAttention`, `CausalMultiHeadAttention`) はカーネル内で未来位置をマスクし、k と v は q より多くの位置を持てます — 自己回帰モデルのプロンプトプレフィルとチャンクデコードのパターンです — のでマスクテンソルも作られません。`CausalMultiHeadAttention` は 1 回の融合 flash-attention 風ディスパッチ (kv タイル上のオンライン softmax、ヘッド次元 128 まで) で走ります: スコア行列は存在しないので、メモリはシーケンス長によらず q+k+v+出力 のままです。

## デバイス上の量子化重み

`UploadQ8` は `QMatrix` を u32 あたり int8 重み 4 つでパックし、`gpu.QMatrix.MatMul` がレジスタ内で逆量子化するので、デコード matvec の転送は f32 の 1/4 バイトになります。`UploadQ4` は int4 の双子に同じことをするので、`-q4 -gpu` は int8 では収まらないモデルを動かせます。

トランスフォーマーのデコード 1 ステップの残りも揃っています — `RMSNorm`、インプレース `RoPE`、`Add`、`SiluMul`、`GroupedCausalAttention` (クエリよりヘッド数の少ない KV キャッシュ)、常駐キャッシュに新しい k/v 行を追記する `CopyRowsInto` — なので `tensai run -q8 -gpu` は全ブロックをデバイス上で実行し、トークンごとに戻ってくるのは隠れ状態だけです。`BeginBatch`/`Flush` は 1 トークン分のディスパッチを 1 回のサブミッションに記録し、解放された中間バッファはバッファプールで再利用されます。

## 学習: アクセラレータフック

ここまでの積はすべて推論の形です。学習では 1 つの matmul につきさらに 2 つ増えます — 入力側の勾配 `grad * w^T` と重みの勾配 `x^T * grad` で、常駐テンソルではそれぞれ `MatMulT` と `MatMulTN` です。

`Device` は `tensai.Accelerator` を満たすので、1 行で、どのパッケージのものであれ CPU が回すはずだった積を GPU に回せます:

```go
dev, err := gpu.Open(gpu.HighPerformance)
if err == nil {
	defer dev.Close()
	tensai.UseAccelerator(dev) // 止めるときは tensai.UseAccelerator(nil)
}
```

デバイスに送られるのは `tensai.DefaultAcceleratorThreshold` (4e8 積和) 以上の積だけです。それより小さいものは AVX2 カーネルの方が速く、往復のぶんだけ損をします。境界を変えるには `tensai.UseAcceleratorThreshold` を使います。バックエンドがエラーを返した積は CPU で実行されるので、アクセラレーションが答えを変えることはありません — 変わるのは所要時間だけです。

学習ステップの他の部分は動きません。活性も勾配も optimizer もホスト側に残るので、高速化された積はそのつどオペランドをアップロードし結果をダウンロードします。これが上限を決めていて、しきい値がこの値である理由でもあります。`-tags wgpu24` 経由の AMD 780M で、射影 2 つのブロック (`x @ w1 -> GELU -> @ w2 -> MSE`、積は 6 つ) の autograd 1 ステップを測ると:

| モデル幅 | CPU (AVX2) | デバイス使用 | |
|---|---|---|---|
| 512 | 31.3ms | 32.4ms | しきい値未満なので変化なし |
| 1024 | 176ms | 145ms | 1.22x |
| 2048 | 1367ms | 955ms | 1.43x |

同じ積を両オペランド常駐で回すと CPU の 2.5〜2.8 倍速いので、残りの差の大半は転送です。これを詰めるには学習グラフ全体をデバイスに置く必要があり、逆伝播が触る要素ごとの演算・活性化・正規化のカーネルが要ります — 推論用の一式ではまだ足りていません。

ここまでが積です。逆伝播の残りも、推論では単独では要らなかった常駐カーネルとして揃っています:

```go
h, _ := gx.MatMul(gw)               // 順伝播
a, _ := h.Activate(gpu.ActGELU)     // 戻りは h.ActivateGrad(gpu.ActGELU, grad)
s, _ := a.Binary(gpu.OpMul, gscale) // add, sub, mul, div。短いオペランドは繰り返す
db, _ := gdelta.SumCols()           // バッチにブロードキャストした行の勾配
gw.AdamStep(ggrad, gm, gv, lr, b1, b2, rc1, rc2, eps, 0)
```

`Activate` と `ActivateGrad` は ReLU・tanh・sigmoid・GELU に対応し、CPU カーネルと同じ式です — ここの GELU は誤差関数版で、推論経路が FFN に融合している tanh 近似ではありません — なので学習の途中でデバイスとホストを行き来できます。`AdamStep` も `optim` のカーネルと一致し、モーメントも含めて同じです。

## クロスオーバーの測定

`_example/wgpu -sweep` はサイズの階段を上りながら、GPU が CPU カーネルを追い越す地点に印を付けます。CPU 側はパッケージの他の部分と同じ `dotRows` カーネルなので、サンプルを 2 回ビルドすればポータブル Go、AVX2、GPU の 2 つの使用パターンを比較できます:

```bash
GOEXPERIMENT=nosimd go build -tags wgpu -o wgpu-nosimd ./_example/wgpu
GOEXPERIMENT=simd   go build -tags wgpu -o wgpu-simd   ./_example/wgpu
./wgpu-nosimd -sweep && ./wgpu-simd -sweep
```

Ryzen の iGPU (AMD Radeon 780M、ネイティブ Windows、AVX2 CPU カーネル) では、レジスタタイリングカーネルが階段の全段を GPU 側に載せます:

```
             shape                   MFLOP   gpu+xfer   resident        cpu   res/cpu
mnist dense  1x100x784@784x128        20.1     1.51ms      597µs      652µs     1.09x
mnist conv2  1x19600x72@72x16         45.2    1.388ms      763µs    2.354ms     3.09x
tiny         1x128x128@128x128         4.2      432µs      302µs      410µs     1.36x
small        1x512x512@512x512       268.4    1.331ms    1.216ms    6.865ms     5.65x
medium       8x512x512@512x512      2147.5    8.053ms    6.297ms    71.56ms    11.36x
large        32x512x512@512x512     8589.9    86.71ms   28.856ms  266.277ms     9.23x
huge         64x512x512@512x512    17179.9  116.374ms   62.128ms  566.726ms     9.12x
```

クロスオーバーは CPU カーネル、GPU ドライバ、転送パターンで動きます — WSL2 内の dozen のような変換レイヤー越しでは比率はほぼ同等〜3 倍に縮み、CPU Vulkan 実装では GPU パスが素直に負けます。出荷するドライバの上で測ってください。

## `-tags wgpu24`: 新しい wgpu-native API と、WSL2 の中の本物の GPU

`-tags wgpu24` は同じ `gpu.Open` API を刷新された wgpu-native C API に対してビルドします — **v29 系**のリリースバイナリと組み合わせてください。新 API の収穫は `WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter` で、非準拠の Vulkan ドライバを見えるようにします。具体的には: Mesa の dozen (Vulkan-on-D3D12) は WSL2 内で本物のホスト GPU を公開しますが、v22 API はそれを非準拠として隠して lavapipe にフォールバックします。wgpu24 ビルドなら届きます:

```bash
VK_DRIVER_FILES=/path/to/dzn_icd.json \
TENSAI_WGPU_LIB=$PWD/wgpu29/lib/libwgpu_native.so \
    go run -tags wgpu24 ./_example/wgpu   # adapter: Microsoft Direct3D12 (AMD Radeon(TM) Graphics)
```

両方のタグを付けると `wgpu24` が勝ちます。そしてこのような環境では、その選択がすべてを決めます — どのアダプタに届くかが決まるからです。WSL2 内の dozen 越しに Radeon 780M で測ると、v22 ビルドは llvmpipe に落ちます:

| | `-tags wgpu` (v22) | `-tags wgpu24` (v29) |
|---|---|---|
| 到達したアダプタ | llvmpipe (ソフトウェア) | Microsoft Direct3D12 (Radeon 780M) |
| f32 `32x512x512@512x512`、入力常駐 | 461.7ms | 69.5ms |
| int8 プレフィル / デコード | 27.0 / 6.0 t/s | 1801 / 27.1 t/s |

`Makefile` が `wgpu24` でビルドするのはこのためです。この差はどちらのビルドがどのハードウェアに届くかであって、同じアダプタ上で 2 つのライブラリを比較した結果ではありません — 準拠ドライバが両方から見える環境なら、どちらでも構いません。

どちらの束縛でビルドするかは変数なので、片方と相性の悪いドライバに当たっても
ファイルを編集せずに切り替えられます:

```bash
make build              # 既定の wgpu24
make build WGPU=wgpu    # v22 の束縛
```

`install` と `cross` も同じ変数を見ます。バイナリとライブラリは対にしてください —
`wgpu24` ビルドには v29 系の `libwgpu_native`、`wgpu` ビルドには v22 が要ります。
バイナリの隣にあるもの以外を読ませたいときは `TENSAI_WGPU_LIB` で指定します。
リリースは両方を配っているので、片方と相性の悪いドライバに当たっても Go を
入れ直す必要はありません — `tensai_*` が `wgpu24` ビルド、`tensai-wgpu22_*` が
もう一方で、中のバイナリ名はどちらも `tensai` です。
