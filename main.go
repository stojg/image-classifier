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

func ImportImages(filename string) ([]CIFAR10Image, error) {
	var images = make([]CIFAR10Image, 0)
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
	return images, nil

}

func main() {
	var trainingSet = make([]CIFAR10Image, 0)

	trainingFiles, err := filepath.Glob("./data/data_batch_*")
	if err != nil {
		log.Printf("Error when trying to find training data: %s", err)
		os.Exit(1)
	}

	for _, dataFile := range trainingFiles {
		images, err := ImportImages(dataFile)
		log.Printf("Importing training data from %s\n", dataFile)
		if err != nil {
			if err != io.EOF {
				log.Printf("Error during import %s", err)
			}

		}
		trainingSet = append(trainingSet, images...)
		log.Printf("%d images imported\n", len(images))

	}
	log.Printf("Totally %d images imported\n", len(trainingSet))
}
