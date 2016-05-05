package main

import (
	"bytes"
	"io"
	"log"
	"os"
	"path/filepath"
)

type ImageSet []CIFAR10Image

func (c ImageSet) asTrainingSet() [][][]float64 {
	data := make([][][]float64, len(c))
	for i := 0; i < len(c); i++ {
		data[i] = make([][]float64, 2)

		data[i][0] = make([]float64, len(c[i].raw))
		copy(data[i][0], c[i].raw)

		data[i][1] = make([]float64, 1)
		data[i][1][0] = float64(c[i].label)
	}
	return data
}

func (c ImageSet) asTestSet() [][]float64 {
	data := make([][]float64, len(c))
	for i := 0; i < len(c); i++ {
		data[i] = make([]float64, len(c[i].raw))
		copy(data[i], c[i].raw)
	}
	return data
}

func loadCIFAR10(pattern string) (ImageSet, []float64) {
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

	var labels []float64
	for _, img := range set {
		labels = append(labels, float64(img.label))
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
}
