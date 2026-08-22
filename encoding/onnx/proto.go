package onnx

// A minimal protocol-buffers writer — varints, fixed32, and
// length-delimited fields are all ONNX needs — so the encoder stays
// dependency-free like the rest of tensai.

import (
	"encoding/binary"
	"math"
)

type pbuf struct {
	b []byte
}

func (p *pbuf) varint(v uint64) {
	p.b = binary.AppendUvarint(p.b, v)
}

func (p *pbuf) key(field, wire int) {
	p.varint(uint64(field)<<3 | uint64(wire))
}

// Int emits an int64 field (wire type 0).
func (p *pbuf) Int(field int, v int64) {
	p.key(field, 0)
	p.varint(uint64(v))
}

// Float emits a float field (wire type 5).
func (p *pbuf) Float(field int, f float32) {
	p.key(field, 5)
	p.b = binary.LittleEndian.AppendUint32(p.b, math.Float32bits(f))
}

// Bytes emits a length-delimited field (wire type 2).
func (p *pbuf) Bytes(field int, b []byte) {
	p.key(field, 2)
	p.varint(uint64(len(b)))
	p.b = append(p.b, b...)
}

// Str emits a string field.
func (p *pbuf) Str(field int, s string) {
	p.Bytes(field, []byte(s))
}

// Msg emits a nested message built by fn.
func (p *pbuf) Msg(field int, fn func(*pbuf)) {
	var sub pbuf
	fn(&sub)
	p.Bytes(field, sub.b)
}

// Ints emits a packed repeated int64 field.
func (p *pbuf) Ints(field int, vs []int64) {
	var sub pbuf
	for _, v := range vs {
		sub.varint(uint64(v))
	}
	p.Bytes(field, sub.b)
}
