package main

import (
	"bytes"
	"io"
	"log"
	"os"
	"path/filepath"
)

type ImageSet []CIFAR10Image

func (c ImageSet) Shape(index int) int {
	return len(c)
}

func (c ImageSet) Reshape(newshape, size int) []byte {
	result := make([]byte, 0)
	for _, img := range c {
		result = append(result, img.data...)
	}

	return result
}

func loadCIFAR10(pattern string) (ImageSet, []byte) {
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

	labels := make([]byte, 0)
	for _, img := range set {
		labels = append(labels, img.label)
	}

	return set, labels
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
	return images, nil
}
