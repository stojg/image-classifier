package main

import (
	"image"
	"image/png"
	"os"
)

type CIFAR10Image struct {
	label byte
	data  []byte
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

func (i *CIFAR10Image) Write(b []byte) (n int, err error) {
	i.label = b[0]
	i.data = make([]byte, len(b)-1)
	copied := copy(i.data, b[1:])
	return copied, nil
}
