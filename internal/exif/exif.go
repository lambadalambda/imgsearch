// Package exif reads the two EXIF fields imgsearch cares about, orientation
// and the capture time, from JPEG files without pulling in a full EXIF
// library. Anything it does not understand is ignored rather than failing.
package exif

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"strings"
	"time"
)

// Info is what Parse extracts. A zero Orientation means "not present".
type Info struct {
	// Orientation is the EXIF orientation tag (1..8), or 0 when absent.
	Orientation int
	// CapturedAt is DateTimeOriginal (falling back to DateTimeDigitized and
	// DateTime). It carries no zone information; the zero value means absent.
	CapturedAt time.Time
}

// SwapsDimensions reports whether the orientation rotates the image by 90°,
// so the displayed width and height are the stored ones swapped.
func (i Info) SwapsDimensions() bool {
	return i.Orientation >= 5 && i.Orientation <= 8
}

const (
	tagOrientation        = 0x0112
	tagDateTime           = 0x0132
	tagExifIFDPointer     = 0x8769
	tagDateTimeOriginal   = 0x9003
	tagDateTimeDigitized  = 0x9004
	typeShort             = 3
	typeLong              = 4
	typeASCII             = 2
	maxSegmentScanBytes   = 4 << 20
	exifDateTimeLayout    = "2006:01:02 15:04:05"
	markerSOI             = 0xD8
	markerAPP1            = 0xE1
	markerSOS             = 0xDA
	markerEOI             = 0xD9
	exifHeader            = "Exif\x00\x00"
	minTIFFHeaderLength   = 8
	ifdEntryLength        = 12
	maxIFDEntriesAccepted = 512
)

// Parse reads EXIF from a JPEG stream. Non-JPEG input and JPEGs without an
// APP1 EXIF segment return an empty Info and no error; only I/O failures
// are reported.
func Parse(r io.Reader) (Info, error) {
	br := bufio.NewReader(io.LimitReader(r, maxSegmentScanBytes))
	var soi [2]byte
	if _, err := io.ReadFull(br, soi[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return Info{}, nil
		}
		return Info{}, err
	}
	if soi[0] != 0xFF || soi[1] != markerSOI {
		return Info{}, nil
	}
	for {
		marker, err := readMarker(br)
		if err != nil {
			return Info{}, ignoreEOF(err)
		}
		if marker == markerSOS || marker == markerEOI {
			return Info{}, nil
		}
		var lengthBytes [2]byte
		if _, err := io.ReadFull(br, lengthBytes[:]); err != nil {
			return Info{}, ignoreEOF(err)
		}
		length := int(binary.BigEndian.Uint16(lengthBytes[:]))
		if length < 2 {
			return Info{}, nil
		}
		payload := make([]byte, length-2)
		if _, err := io.ReadFull(br, payload); err != nil {
			return Info{}, ignoreEOF(err)
		}
		if marker == markerAPP1 && strings.HasPrefix(string(payload), exifHeader) {
			return ParseTIFF(payload[len(exifHeader):]), nil
		}
	}
}

func ignoreEOF(err error) error {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return nil
	}
	return err
}

// readMarker skips fill bytes and returns the next marker code.
func readMarker(br *bufio.Reader) (byte, error) {
	for {
		b, err := br.ReadByte()
		if err != nil {
			return 0, err
		}
		if b != 0xFF {
			continue
		}
		for {
			m, err := br.ReadByte()
			if err != nil {
				return 0, err
			}
			if m == 0xFF {
				continue
			}
			return m, nil
		}
	}
}

// ParseTIFF decodes an EXIF TIFF blob (the APP1 payload after "Exif\0\0").
// Malformed input yields whatever fields were readable.
func ParseTIFF(data []byte) Info {
	var info Info
	if len(data) < minTIFFHeaderLength {
		return info
	}
	var order binary.ByteOrder
	switch string(data[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return info
	}
	if order.Uint16(data[2:4]) != 42 {
		return info
	}
	ifd0 := int(order.Uint32(data[4:8]))
	var dateTime, dateTimeOriginal, dateTimeDigitized string
	var exifIFD int
	walkIFD(data, order, ifd0, func(tag uint16, typ uint16, count uint32, value []byte) {
		switch tag {
		case tagOrientation:
			if typ == typeShort && count >= 1 {
				info.Orientation = int(order.Uint16(value[:2]))
			}
		case tagDateTime:
			dateTime = asciiValue(data, order, typ, count, value)
		case tagExifIFDPointer:
			if typ == typeLong && count >= 1 {
				exifIFD = int(order.Uint32(value[:4]))
			}
		}
	})
	if exifIFD > 0 {
		walkIFD(data, order, exifIFD, func(tag uint16, typ uint16, count uint32, value []byte) {
			switch tag {
			case tagDateTimeOriginal:
				dateTimeOriginal = asciiValue(data, order, typ, count, value)
			case tagDateTimeDigitized:
				dateTimeDigitized = asciiValue(data, order, typ, count, value)
			}
		})
	}
	if info.Orientation < 1 || info.Orientation > 8 {
		info.Orientation = 0
	}
	for _, candidate := range []string{dateTimeOriginal, dateTimeDigitized, dateTime} {
		if t, ok := ParseDateTime(candidate); ok {
			info.CapturedAt = t
			break
		}
	}
	return info
}

// walkIFD calls fn for every entry of the IFD at offset with the entry's
// 4-byte value field (inline value or offset).
func walkIFD(data []byte, order binary.ByteOrder, offset int, fn func(tag uint16, typ uint16, count uint32, value []byte)) {
	if offset < 0 || offset+2 > len(data) {
		return
	}
	entries := int(order.Uint16(data[offset : offset+2]))
	if entries > maxIFDEntriesAccepted {
		entries = maxIFDEntriesAccepted
	}
	pos := offset + 2
	for i := 0; i < entries; i++ {
		if pos+ifdEntryLength > len(data) {
			return
		}
		entry := data[pos : pos+ifdEntryLength]
		fn(order.Uint16(entry[0:2]), order.Uint16(entry[2:4]), order.Uint32(entry[4:8]), entry[8:12])
		pos += ifdEntryLength
	}
}

// asciiValue resolves an ASCII entry, inline when it fits in 4 bytes.
func asciiValue(data []byte, order binary.ByteOrder, typ uint16, count uint32, value []byte) string {
	if typ != typeASCII || count == 0 || count > 256 {
		return ""
	}
	var raw []byte
	if count <= 4 {
		raw = value[:count]
	} else {
		start := int(order.Uint32(value[:4]))
		end := start + int(count)
		if start < 0 || end > len(data) {
			return ""
		}
		raw = data[start:end]
	}
	return strings.TrimRight(string(raw), "\x00 ")
}

// ParseDateTime parses the EXIF "YYYY:MM:DD HH:MM:SS" form. Unset values
// ("0000:00:00 00:00:00", blanks) report false.
func ParseDateTime(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" || strings.HasPrefix(s, "0000") {
		return time.Time{}, false
	}
	t, err := time.Parse(exifDateTimeLayout, s)
	if err != nil {
		return time.Time{}, false
	}
	return t, true
}

// SQLiteTime renders a capture time in the "YYYY-MM-DD HH:MM:SS" form the
// database uses for created_at, so the two sort together.
func SQLiteTime(t time.Time) string {
	return t.Format("2006-01-02 15:04:05")
}
