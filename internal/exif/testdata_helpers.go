package exif

import "encoding/binary"

// BuildAPP1 assembles a little-endian EXIF TIFF blob with the given
// orientation and DateTimeOriginal, wrapped as a JPEG APP1 segment. It is
// exported for tests in other packages that need a rotated fixture.
func BuildAPP1(orientation int, dateTimeOriginal string) []byte {
	tiff := BuildTIFF(orientation, dateTimeOriginal)
	payload := append([]byte(exifHeader), tiff...)
	seg := []byte{0xFF, markerAPP1, 0, 0}
	binary.BigEndian.PutUint16(seg[2:4], uint16(len(payload)+2))
	return append(seg, payload...)
}

// BuildTIFF returns the raw TIFF blob used by BuildAPP1: IFD0 with
// Orientation and an Exif IFD pointer, then an Exif IFD with
// DateTimeOriginal stored out of line.
func BuildTIFF(orientation int, dateTimeOriginal string) []byte {
	order := binary.LittleEndian
	_ = order
	buf := []byte{'I', 'I', 42, 0, 8, 0, 0, 0}
	// IFD0: 2 entries at offset 8: Orientation, ExifIFDPointer; next IFD = 0.
	ifd0Len := 2 + 2*ifdEntryLength + 4
	exifIFDOffset := 8 + ifd0Len
	ifd0 := make([]byte, 0, ifd0Len)
	ifd0 = order.AppendUint16(ifd0, 2)
	ifd0 = appendEntry(ifd0, order, tagOrientation, typeShort, 1, uint32(orientation))
	ifd0 = appendEntry(ifd0, order, tagExifIFDPointer, typeLong, 1, uint32(exifIFDOffset))
	ifd0 = order.AppendUint32(ifd0, 0)
	buf = append(buf, ifd0...)
	// Exif IFD: 1 entry, DateTimeOriginal ASCII (count includes NUL) out of line.
	exifIFDLen := 2 + ifdEntryLength + 4
	dateOffset := exifIFDOffset + exifIFDLen
	exifIFD := make([]byte, 0, exifIFDLen)
	exifIFD = order.AppendUint16(exifIFD, 1)
	exifIFD = appendEntry(exifIFD, order, tagDateTimeOriginal, typeASCII, uint32(len(dateTimeOriginal)+1), uint32(dateOffset))
	exifIFD = order.AppendUint32(exifIFD, 0)
	buf = append(buf, exifIFD...)
	buf = append(buf, []byte(dateTimeOriginal)...)
	buf = append(buf, 0)
	return buf
}

func appendEntry(b []byte, order binary.AppendByteOrder, tag uint16, typ uint16, count uint32, value uint32) []byte {
	b = order.AppendUint16(b, tag)
	b = order.AppendUint16(b, typ)
	b = order.AppendUint32(b, count)
	if typ == typeShort {
		b = order.AppendUint16(b, uint16(value))
		b = order.AppendUint16(b, 0)
		return b
	}
	return order.AppendUint32(b, value)
}

// InsertAPP1 places an APP1 segment right after the SOI marker of a JPEG.
func InsertAPP1(jpeg []byte, app1 []byte) []byte {
	if len(jpeg) < 2 {
		return jpeg
	}
	out := make([]byte, 0, len(jpeg)+len(app1))
	out = append(out, jpeg[:2]...)
	out = append(out, app1...)
	return append(out, jpeg[2:]...)
}
