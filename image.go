package main

import (
	"bytes"
	"image"
	"image/png"
	"io"
	"log"
	"os"
	"path/filepath"
)

// ImageSet is an collection of CIFAR10Images
type ImageSet []CIFAR10Image

func (c ImageSet) asFloatSlices() ([][]float64, [][]byte) {
	x := make([][]float64, len(c))
	y := make([][]byte, len(c))
	for i := 0; i < len(c); i++ {
		x[i] = make([]float64, len(c[i].raw))
		copy(x[i], c[i].raw)
		y[i] = make([]byte, 10)
		if c[i].label == 0 {
		}
		y[i][c[i].label] = 1.0
	}
	return x, y
}

func loadCIFAR10(pattern string) ImageSet {
	var set = make([]CIFAR10Image, 0)
	trainingFiles, err := filepath.Glob(pattern)
	if err != nil {
		log.Printf("error when trying to find training data: %s", err)
		os.Exit(1)
	}
	for _, dataFile := range trainingFiles {
		images, err := imagesFromFile(dataFile)
		log.Printf("importing data from %s\n", dataFile)
		if err != nil {
			if err != io.EOF {

				log.Printf("error during import %s", err)
			}
		}
		set = append(set, images...)
	}

	return set
}

func imagesFromFile(filename string) (ImageSet, error) {
	var images = make(ImageSet, 0)
	f, err := os.Open(filename)
	if err != nil {
		return images, err
	}
	defer f.Close()

	for {
		data := make([]byte, 1+1024*3)
		if _, err := f.Read(data); err != nil {
			return images, err
		}
		image := CIFAR10Image{}
		io.Copy(&image, bytes.NewReader(data))
		images = append(images, image)
	}
}

// CIFAR10Image represent a singular image from the CIFAR10Image set
type CIFAR10Image struct {
	label byte
	data  []byte
	raw   []float64
}

// Decode will return a image.Image that can be converted to a png or what not
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
	i.label = (b[0])
	i.raw = make([]float64, len(b)-1)
	for idx, val := range b[1:] {
		i.raw[idx] = float64(val)
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
