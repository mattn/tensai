package tflite

import "encoding/binary"

// fbBuilder is a minimal FlatBuffers builder, implementing just the subset
// the TFLite schema needs: tables, scalar fields, offset fields, vectors,
// and strings. It follows the reference implementation's design: the buffer
// is filled back to front, and offsets are measured from the end of the
// buffer toward the front.
type fbBuilder struct {
	buf       []byte
	head      int // index of the next write position (moves toward 0)
	minalign  int
	vtable    []int // per-slot object offsets while building a table
	objectEnd int
}

func newFbBuilder() *fbBuilder {
	const initial = 1024
	return &fbBuilder{buf: make([]byte, initial), head: initial, minalign: 1}
}

// offset returns the current position measured from the end of the buffer.
func (b *fbBuilder) offset() int {
	return len(b.buf) - b.head
}

// prep pads so that after writing additionalBytes, the position is aligned
// to size, growing the buffer at the front when needed.
func (b *fbBuilder) prep(size, additionalBytes int) {
	if size > b.minalign {
		b.minalign = size
	}
	alignSize := (^(len(b.buf) - b.head + additionalBytes) + 1) & (size - 1)
	for b.head < alignSize+size+additionalBytes {
		oldBuf := b.buf
		b.buf = make([]byte, len(oldBuf)*2)
		b.head += len(oldBuf)
		copy(b.buf[len(oldBuf):], oldBuf)
	}
	for i := 0; i < alignSize; i++ {
		b.head--
		b.buf[b.head] = 0
	}
}

func (b *fbBuilder) placeByte(v byte) {
	b.head--
	b.buf[b.head] = v
}

func (b *fbBuilder) placeUint16(v uint16) {
	b.head -= 2
	binary.LittleEndian.PutUint16(b.buf[b.head:], v)
}

func (b *fbBuilder) placeUint32(v uint32) {
	b.head -= 4
	binary.LittleEndian.PutUint32(b.buf[b.head:], v)
}

func (b *fbBuilder) prependByte(v byte)     { b.prep(1, 0); b.placeByte(v) }
func (b *fbBuilder) prependUint16(v uint16) { b.prep(2, 0); b.placeUint16(v) }
func (b *fbBuilder) prependUint32(v uint32) { b.prep(4, 0); b.placeUint32(v) }
func (b *fbBuilder) prependInt32(v int32)   { b.prependUint32(uint32(v)) }

// prependUOffset writes a forward reference to a previously built object.
func (b *fbBuilder) prependUOffset(off int) {
	b.prep(4, 0)
	b.placeUint32(uint32(b.offset() - off + 4))
}

// startObject begins a table with the given number of fields.
func (b *fbBuilder) startObject(numFields int) {
	if cap(b.vtable) < numFields {
		b.vtable = make([]int, numFields)
	} else {
		b.vtable = b.vtable[:numFields]
		for i := range b.vtable {
			b.vtable[i] = 0
		}
	}
	b.objectEnd = b.offset()
}

func (b *fbBuilder) slot(id int) {
	b.vtable[id] = b.offset()
}

// Scalar slots are omitted when the value equals the schema default.

func (b *fbBuilder) byteSlot(id int, v, def byte) {
	if v == def {
		return
	}
	b.prependByte(v)
	b.slot(id)
}

func (b *fbBuilder) int32Slot(id int, v, def int32) {
	if v == def {
		return
	}
	b.prependInt32(v)
	b.slot(id)
}

func (b *fbBuilder) uint32Slot(id int, v, def uint32) {
	if v == def {
		return
	}
	b.prependUint32(v)
	b.slot(id)
}

func (b *fbBuilder) float32Slot(id int, v, def float32) {
	if v == def {
		return
	}
	b.prep(4, 0)
	b.placeUint32(fbits(v))
	b.slot(id)
}

func (b *fbBuilder) uoffsetSlot(id, off int) {
	if off == 0 {
		return
	}
	b.prependUOffset(off)
	b.slot(id)
}

// endObject writes the table's vtable and returns the table offset.
func (b *fbBuilder) endObject() int {
	// Placeholder for the soffset to the vtable.
	b.prep(4, 0)
	b.placeUint32(0)
	objectOffset := b.offset()

	// Trim trailing empty slots.
	n := len(b.vtable)
	for n > 0 && b.vtable[n-1] == 0 {
		n--
	}
	for i := n - 1; i >= 0; i-- {
		var off uint16
		if b.vtable[i] != 0 {
			off = uint16(objectOffset - b.vtable[i])
		}
		b.prependUint16(off)
	}
	b.prependUint16(uint16(objectOffset - b.objectEnd)) // table size
	b.prependUint16(uint16((n + 2) * 2))                // vtable size
	vtableOffset := b.offset()

	// Patch the placeholder: reader computes vtable = table - soffset.
	pos := len(b.buf) - objectOffset
	binary.LittleEndian.PutUint32(b.buf[pos:], uint32(int32(vtableOffset)-int32(objectOffset)))
	return objectOffset
}

// startVector prepares a vector of count elements of elemSize bytes. Pass
// forceAlign > elemSize for extra alignment (e.g. buffer payloads).
func (b *fbBuilder) startVector(elemSize, count, forceAlign int) {
	b.prep(4, elemSize*count)
	if forceAlign > 0 {
		b.prep(forceAlign, elemSize*count)
	}
}

func (b *fbBuilder) endVector(count int) int {
	b.placeUint32(uint32(count))
	return b.offset()
}

// int32Vector builds a [int] / [uint] vector.
func (b *fbBuilder) int32Vector(vals []int32) int {
	b.startVector(4, len(vals), 0)
	for i := len(vals) - 1; i >= 0; i-- {
		b.placeUint32(uint32(vals[i]))
	}
	return b.endVector(len(vals))
}

// offsetVector builds a vector of references to previously built objects.
func (b *fbBuilder) offsetVector(offs []int) int {
	b.startVector(4, len(offs), 0)
	for i := len(offs) - 1; i >= 0; i-- {
		b.prependUOffset(offs[i])
	}
	return b.endVector(len(offs))
}

// byteVector builds a [ubyte] vector with the given forced alignment.
func (b *fbBuilder) byteVector(data []byte, forceAlign int) int {
	b.startVector(1, len(data), forceAlign)
	b.head -= len(data)
	copy(b.buf[b.head:], data)
	return b.endVector(len(data))
}

// createString builds a null-terminated string.
func (b *fbBuilder) createString(s string) int {
	b.prep(4, len(s)+1)
	b.placeByte(0)
	b.head -= len(s)
	copy(b.buf[b.head:], s)
	return b.endVector(len(s))
}

// finish writes the root reference and the 4-byte file identifier and
// returns the completed buffer.
func (b *fbBuilder) finish(root int, identifier string) []byte {
	b.prep(b.minalign, 4+4)
	b.head -= 4
	copy(b.buf[b.head:], identifier)
	b.prependUOffset(root)
	return b.buf[b.head:]
}
