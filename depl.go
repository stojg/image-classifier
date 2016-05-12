package main

import (
	"database/sql"
	_ "github.com/go-sql-driver/mysql"
	"log"
	"time"
	"fmt"
	"os"
)

// @todo: add features for:
// - time since last release to the same env
// - same sha as last deployment to the same env
// - same deployer as last deploy
// - add fast / slow deployment
// - add type of backend (rainforest/capistrano)
// Convert the whole object back and forth to/from  a byte array
type Deployment struct {
	id             int
	className      string
	created        time.Time
	lastEdited     time.Time
	sha            sql.NullString
	environmentID  int
	deployerID     int
	status         string
	projectID      int
	branch         sql.NullString
	prevDeployment *Deployment
}

func (d *Deployment) Update(rows *sql.Rows) error {
	d.prevDeployment = nil
	return rows.Scan(
		&d.id,
		&d.className,
		&d.created,
		&d.lastEdited,
		&d.sha,
		&d.environmentID,
		&d.deployerID,
		&d.status,
		&d.branch,
	)
}

func (d *Deployment) asData() ([]float64, []byte) {
	var firstDeploy bool = true
	var sameShaAsPrevious bool = false

	if d.prevDeployment != nil {
		firstDeploy = false
		sameShaAsPrevious = (d.sha.String == d.prevDeployment.sha.String)
	}

	x := make([]float64, 0)
	x = append(x, float64(d.created.Weekday()))
	x = append(x, float64(d.created.Hour()))
	x = append(x, d.booleanAsFloat(sameShaAsPrevious))
	x = append(x, d.booleanAsFloat(firstDeploy))

	var daysSincePrevious float64 = 356
	if d.prevDeployment != nil {
		daysSincePrevious = float64(d.created.Sub(d.prevDeployment.created).Hours() / 24)
	}
	x = append(x, daysSincePrevious)

	// encode env id


	// 8

	a := make([]byte, 16)
	n := uint8(5)
	k := uint8(0)
	a[0] = (n & (1 << k)) >> k

	fmt.Printf("%v", (uint8(n) >> 0) & 0xFF)

//	fmt.Printf("%v\n", strconv.FormatInt(int64(d.environmentID), 2))
//	fmt.Printf("%v\n", strconv.FormatInt(int64(d.deployerID), 2))
	os.Exit(1)

//	for i := 0; i<610; i++ {
//		if d.environmentID == i {
//			result[0] = append(result[0], 1)
//		} else {
//			result[0] = append(result[0], 0)
//		}
//	}
//
//	for i := 0; i<600; i++ {
//		if d.deployerID == i {
//			result[0] = append(result[0], 1)
//		} else {
//			result[0] = append(result[0], 0)
//		}
//	}

	y := make([]byte, 2)
	if d.statusAsFloat() > 0.5 {
		y[1] = 1
	} else {
		y[0] = 1
	}

	return x, y
}

func (d *Deployment) statusAsFloat() float64 {
	switch d.status {
	case statusFailed:
		return 0
	case statusSuccess:
		return 1
	default:
		return 0
	}
}

func (d *Deployment) booleanAsFloat(val bool) float64 {
	if val {
		return 1
	}
	return 0
}

type DeployList []*Deployment

func (l DeployList) Link() {
	envList := make(map[int][]*Deployment, 0)

	for _, dep := range l {
		var _ []*Deployment
		if _, ok := envList[dep.environmentID]; !ok {
			envList[dep.environmentID] = make([]*Deployment, 0)
		}
		envList[dep.environmentID] = append(envList[dep.environmentID], dep)
	}

	// @todo sort list by descending order so we don't have to rely on the
	// sql ORDER By

	for envId, environments := range envList {
		for idx := range environments {
			if idx == 0 {
				continue
			}
			envList[envId][idx].prevDeployment = envList[envId][idx-1]

		}
	}
}

const (
	statusFailed  = "Failed"
	statusSuccess = "Finished"
)

func loadDeployData() (x [][]float64, y [][]byte, err error) {

	connection := "root:@tcp(127.0.0.1:3306)/deploynaut?timeout=500ms&parseTime=true&loc=Pacific%2FAuckland"
	db, err := sql.Open("mysql", connection)
	if err != nil {
		return x, y, err
	}
	defer db.Close()

	log.Printf("Connected to database\n")
	rows, err := db.Query("SELECT ID, ClassName, Created, LastEdited, SHA, EnvironmentID, DeployerID, Status, Branch from DNDeployment ORDER BY EnvironmentID, ID")
	if err != nil {
		return x, y, err
	}
	defer rows.Close()

	var deployments DeployList
	for rows.Next() {
		deployment := &Deployment{}
		if err := deployment.Update(rows); err != nil {
			log.Fatal(err)
		}
		deployments = append(deployments, deployment)
	}

	log.Printf("Linking deployments to previous deployment")
	deployments.Link()

	x = make([][]float64, len(deployments))
	y = make([][]byte, len(deployments))
	for i, deploy := range deployments {
		dX, dY := deploy.asData()
		x[i] = append(x[i], dX...)
		y[i] = append(y[i], dY...)
	}

	return x, y, err
}
