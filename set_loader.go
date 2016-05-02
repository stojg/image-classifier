package main

import (
	"bytes"
	"io"
	"log"
	"os"
	"path/filepath"
)

func getImageSet(pattern string) []CIFAR10Image {
	var set = make([]CIFAR10Image, 0)
	trainingFiles, err := filepath.Glob(pattern)
	if err != nil {
		log.Printf("error when trying to find training data: %s", err)
		os.Exit(1)
	}
	for _, dataFile := range trainingFiles {
		images, err := imagesFromFile(dataFile)
		log.Printf("importing training data from %s\n", dataFile)
		if err != nil {
			if err != io.EOF {
				log.Printf("error during import %s", err)
			}
		}
		set = append(set, images...)
	}
	return set
}

func imagesFromFile(filename string) ([]CIFAR10Image, error) {
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
