package main

import (
	"log"
)

func main() {
	set := getImageSet("./data/data_batch_*")
	log.Printf("%d images in training set\n", len(set))
	//log.Printf("%d\n", set[1].label)
	//toPNG(trainingSet[1], "output.png")
}
