// Copyright 2026 The tensai Authors. All rights reserved.

//go:build goexperiment.simd && arm64 && go1.27

#include "textflag.h"

// sdotTile32 accumulates one 32-column tile of the int8 matvec.
//
// SDOT multiplies four signed bytes against four more and adds the four
// products into a 32-bit lane, which is the shape of this layout: sixteen
// weight bytes hold four columns four rows deep, and one instruction
// closes all four columns. Go's simd/archsimd has no dot product on
// arm64 and the assembler knows only the SVE spelling, which Apple
// silicon does not implement, so the four instruction words go in by
// hand. They are sdot v8..v15.4s, v0..v7.16b, v16.16b, checked against
// the assembler's own encoding.
//
// The activations arrive in the loader's offset-binary form, so the
// broadcast quad takes one vector subtract to become the signed bytes
// SDOT wants; that is cheaper than packing a signed copy of the row.
//
// func sdotTile32(acc *int32, tile *int8, xu *uint8, quads int)
TEXT ·sdotTile32(SB), NOSPLIT|NOFRAME, $0-32
	MOVD	acc+0(FP), R0
	MOVD	tile+8(FP), R1
	MOVD	xu+16(FP), R2
	MOVD	quads+24(FP), R3

	MOVD	$64, R5
	VDUP	R5, V17.B16

	VEOR	V8.B16, V8.B16, V8.B16
	VEOR	V9.B16, V9.B16, V9.B16
	VEOR	V10.B16, V10.B16, V10.B16
	VEOR	V11.B16, V11.B16, V11.B16
	VEOR	V12.B16, V12.B16, V12.B16
	VEOR	V13.B16, V13.B16, V13.B16
	VEOR	V14.B16, V14.B16, V14.B16
	VEOR	V15.B16, V15.B16, V15.B16

loop:
	MOVWU.P	4(R2), R4
	VDUP	R4, V16.S4
	VSUB	V17.B16, V16.B16, V16.B16
	VLD1.P	64(R1), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P	64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]

	WORD	$0x4E909408 // sdot v8.4s, v0.16b, v16.16b
	WORD	$0x4E909429 // sdot v9.4s, v1.16b, v16.16b
	WORD	$0x4E90944A // sdot v10.4s, v2.16b, v16.16b
	WORD	$0x4E90946B // sdot v11.4s, v3.16b, v16.16b
	WORD	$0x4E90948C // sdot v12.4s, v4.16b, v16.16b
	WORD	$0x4E9094AD // sdot v13.4s, v5.16b, v16.16b
	WORD	$0x4E9094CE // sdot v14.4s, v6.16b, v16.16b
	WORD	$0x4E9094EF // sdot v15.4s, v7.16b, v16.16b

	SUB	$1, R3
	CBNZ	R3, loop

	VST1.P	[V8.S4, V9.S4, V10.S4, V11.S4], 64(R0)
	VST1	[V12.S4, V13.S4, V14.S4, V15.S4], (R0)
	RET
