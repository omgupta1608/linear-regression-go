package LinearRegression

import (
	"log"
	"math"
	"math/rand"
	"time"
)

var (
	DEFAULT_ALPHA      float64 = 0.01
	DEFAULT_ITERATIONS int     = 2
)

type LinearRegressionModel struct {
	_Coeffs    []float64
	_Intercept float64
}

func gradientDescent(X [][]float64, Y []float64, alpha float64, iterations int, betas []float64) []float64 {
	m := float64(len(Y))
	for i := 0; i < iterations; i++ {
		// Initialize gradients
		gradients := make([]float64, len(betas))
		for j := 0; j < len(Y); j++ {
			prediction := betas[0]
			for k := 0; k < len(X[j]); k++ {
				prediction += X[j][k] * betas[k+1]
			}
			err := prediction - Y[j]

			gradients[0] += err
			for k := 0; k < len(X[j]); k++ {
				gradients[k+1] += err * X[j][k]
			}
		}
		for k := 0; k < len(betas); k++ {
			if math.IsNaN(gradients[k]) {
				log.Fatalln("gradients gone NaN")
			}
			betas[k] -= (alpha / m) * gradients[k]
		}
	}

	return betas
}

func Fit(X [][]float64, Y []float64, alpha *float64, iterations *int) (model LinearRegressionModel) {
	if alpha == nil {
		alpha = &DEFAULT_ALPHA
	}
	if iterations == nil {
		iterations = &DEFAULT_ITERATIONS
	}
	if len(X) != len(Y) {
		log.Fatalln("X and Y should have equal lengths")
	}
	model._Coeffs = make([]float64, len(X[0]))
	model._Intercept = 0.0

	betas := gradientDescent(X, Y, *alpha, *iterations, append([]float64{model._Intercept}, model._Coeffs...))
	model._Intercept = betas[0]
	model._Coeffs = betas[1:]
	return
}

func (model LinearRegressionModel) Predict(X []float64) float64 {
	if len(X) != len(model._Coeffs) {
		log.Fatalln("error: X has invalid shape")
	}
	prediction := model._Intercept
	for k := 0; k < len(model._Coeffs); k++ {
		prediction += X[k] * model._Coeffs[k]
	}
	return prediction
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
