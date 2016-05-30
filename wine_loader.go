package main

import (
	"encoding/csv"
	"io"
	"os"
	"strconv"
)

func wineLoader(file string) ([][]float64, [][]float64, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	csvr := csv.NewReader(f)

	var data [][]float64
	var classes [][]float64
	for {
		row, err := csvr.Read()
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return data, classes, err
		}

		class := make([]float64, 3)
		var classVal float64
		if classVal, err = strconv.ParseFloat(row[0], 64); err != nil {
			return data, classes, err
		}
		class[int(classVal)-1] = 1

		classes = append(classes, class)
		values := make([]float64, len(row))
		for i := 1; i < len(row)-1; i++ {
			var val float64
			if val, err = strconv.ParseFloat(row[i], 64); err != nil {
				return data, classes, err
			}
			values[i] = val

		}
		data = append(data, values)
	}
}
