package main

import (
	"image"
	"image/png"
	"os"
)

type CIFAR10Image struct {
	label dLabel
	data  []byte
	raw   []dValue
}

func (i *CIFAR10Image) Decode() (image.Image, error) {

	img := image.NewRGBA(image.Rectangle{
		Min: image.Point{X: 0, Y: 0},
		Max: image.Point{X: 32, Y: 32},
	})
	// alpha channel
	for j := 0; j < len(i.data)/3; j++ {
		img.Pix[j*4+3] = 255
	}
	for channel := 0; channel < 3; channel++ {
		for j := 0; j < len(i.data)/3; j++ {
			img.Pix[j*4+channel] = i.data[j+channel*1024]
		}
	}

	return img, nil
}

func (i *CIFAR10Image) Write(b []byte) (n int, err error) {
	i.label = dLabel(b[0])
	i.raw = make([]dValue, len(b)-1)
	for idx, val := range b[1:] {
		i.raw[idx] = dValue(val)
	}
	return len(b), nil
}

func toPNG(i CIFAR10Image, location string) error {
	var out *os.File
	var err error
	if out, err = os.Create(location); err != nil {
		return err
	}
	img, _ := i.Decode()
	if err := png.Encode(out, img); err != nil {
		return err
	}
	return nil
}
